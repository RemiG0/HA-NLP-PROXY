import joblib, torch, numpy as np, pathlib
from transformers import AutoTokenizer, AutoModel
import logging
import json

logger = logging.getLogger("ha-nlp-proxy")

MODEL_NAME = "allegro/herbert-base-cased"
_tokenizer = None
_bert = None
_clf_intent = None
_clf_entity = None
_le_intent = None
_le_entity = None

def build_ha_arguments(entity_id: str, prompt: str) -> list[str]:
    """Wnioskuje domenę z tekstu i zwraca listę argumentów JSON dla Home Assistant."""
    if not entity_id:
        return []
        
    if entity_id.startswith("area."):
        raw_area_id = entity_id[5:]
        prompt_lower = prompt.lower()
        
        # Determine target domains and keywords
        target_domains = set()
        keywords = []
        if any(w in prompt_lower for w in ["światł", "lamp", "kinkiet", "light", "bulb"]):
            target_domains.add("light")
            keywords.extend(["światł", "lamp", "kinkiet", "light", "bulb"])
        if any(w in prompt_lower for w in ["rolet", "żaluz", "zasłon", "cover", "blind", "shade"]):
            target_domains.add("cover")
            keywords.extend(["rolet", "żaluz", "zasłon", "cover", "blind", "shade"])
        if any(w in prompt_lower for w in ["odkurz", "sprząt", "vacuum", "robot"]):
            target_domains.add("vacuum")
            keywords.extend(["odkurz", "sprząt", "vacuum", "robot"])
        if any(w in prompt_lower for w in ["wiatrak", "wentylator", "fan"]):
            target_domains.add("fan")
            keywords.extend(["wiatrak", "wentylator", "fan"])
            
        if not target_domains:
            return [json.dumps({"area": raw_area_id}, ensure_ascii=False)]
            
        # Search the database for matching entities in this area
        from db import engine, Entity
        from sqlmodel import Session, select
        matched_entity_ids = []
        with Session(engine) as session:
            entities_in_area = session.exec(select(Entity).where(Entity.area_id == raw_area_id)).all()
            for ent in entities_in_area:
                if ent.domain in target_domains:
                    matched_entity_ids.append(ent.original_name or ent.friendly_name or ent.entity_id)
                    continue
                
                # Intelligent alias/name match for cross-domain devices (e.g. switch acting as light)
                name_lower = (ent.friendly_name or "").lower()
                aliases_lower = (ent.aliases or "").lower()
                if any(kw in name_lower or kw in aliases_lower for kw in keywords):
                    matched_entity_ids.append(ent.original_name or ent.friendly_name or ent.entity_id)
        
        if matched_entity_ids:
            return [json.dumps({"name": eid}, ensure_ascii=False) for eid in matched_entity_ids]
        else:
            # Fallback to the area + domain payload if no specific entities found
            return [json.dumps({"area": raw_area_id, "domain": list(target_domains)}, ensure_ascii=False)]
    else:
        from db import engine, Entity
        from sqlmodel import Session, select
        with Session(engine) as session:
            ent = session.exec(select(Entity).where(Entity.entity_id == entity_id)).first()
            if ent and ent.original_name:
                return [json.dumps({"name": ent.original_name}, ensure_ascii=False)]
            elif ent and ent.friendly_name:
                return [json.dumps({"name": ent.friendly_name}, ensure_ascii=False)]
        return [json.dumps({"name": entity_id}, ensure_ascii=False)]

def load_models():
    global _tokenizer, _bert, _clf_intent, _clf_entity, _le_intent, _le_entity
    
    logger.info("Loading tokenizer and BERT model...")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _bert = AutoModel.from_pretrained(MODEL_NAME).eval()
    
    p = pathlib.Path("models")
    if not p.exists() or not (p / "intent_clf.joblib").exists():
        logger.warning("Local classifier models not found in models/ directory. Run training first.")
        return
        
    logger.info("Loading scikit-learn models...")
    _clf_intent = joblib.load(p / "intent_clf.joblib")
    _clf_entity = joblib.load(p / "entity_clf.joblib")
    _le_intent  = joblib.load(p / "label_enc_intent.joblib")
    _le_entity  = joblib.load(p / "label_enc_entity.joblib")
    logger.info("Models loaded successfully.")

def classify(text: str, threshold: float, return_all: bool = False):
    if _clf_intent is None or _clf_entity is None:
        logger.warning("Classifiers not loaded, falling back immediately.")
        return None, None, 0.0, 0.0

    text = text.lower()
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        emb = _bert(**inputs).last_hidden_state[:, 0, :].squeeze().numpy().reshape(1, -1)

    import numpy as np
    def sigmoid_scale(margins, a=10):
        return 1.0 / (1.0 + np.exp(-margins * a))

    i_logits = _clf_intent.decision_function(emb)[0]
    intent_score = float(np.max(sigmoid_scale(i_logits, a=10)))

    e_logits = _clf_entity.decision_function(emb)[0]
    entity_score = float(np.max(sigmoid_scale(e_logits, a=10)))

    intent    = _le_intent.inverse_transform(_clf_intent.predict(emb))[0]
    entity_id = _le_entity.inverse_transform(_clf_entity.predict(emb))[0]

    if not return_all and (intent_score < threshold or entity_score < threshold):
        return None, None, intent_score, entity_score

    return intent, entity_id, intent_score, entity_score
