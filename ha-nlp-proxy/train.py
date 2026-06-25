from sqlmodel import Session, select
from db import engine, IntentSample, Entity
from transformers import AutoTokenizer, AutoModel
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
import torch, joblib, numpy as np, pathlib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train")

MODEL_NAME = "allegro/herbert-base-cased"
OUT = pathlib.Path("models")
OUT.mkdir(exist_ok=True)

def get_embeddings(sentences, tokenizer, model, batch_size=32):
    embeddings = []
    # Lowercase all sentences for case-insensitive matching
    sentences = [s.lower() for s in sentences]
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            emb = model(**inputs).last_hidden_state[:, 0, :].numpy()
        embeddings.append(emb)
    if not embeddings:
        return np.array([])
    return np.vstack(embeddings)

def train():
    logger.info("Loading HuggingFace models...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    bert = AutoModel.from_pretrained(MODEL_NAME).eval()

    with Session(engine) as session:
        samples = session.exec(select(IntentSample)).all()
        entities = session.exec(select(Entity).where(Entity.enabled == True)).all()

    if not samples:
        logger.error("No intent samples found. Add samples first.")
        return
        
    if not entities:
        logger.warning("No entities found. Add entities to train the entity classifier.")
        
    logger.info(f"Training on {len(samples)} intent samples and {len(entities)} entities.")

    # ── Intent classifier ─────────────────────────────────
    sentences   = [s.sentence for s in samples]
    intent_lbls = [s.intent   for s in samples]

    X = get_embeddings(sentences, tokenizer, bert)

    le_intent = LabelEncoder().fit(intent_lbls)
    clf_intent = LinearSVC(C=1.0, class_weight="balanced", dual=False, max_iter=5000).fit(X, le_intent.transform(intent_lbls))

    # ── Entity classifier ─────────────────────────────────
    if entities:
        ent_sentences = []
        ent_labels = []
        from collections import defaultdict
        
        from db import InflectionRule
        with Session(engine) as session:
            inflection_rules = session.exec(select(InflectionRule)).all()
            
        entity_phrases = defaultdict(set)
        for e in entities:
            base_names = [e.friendly_name]
            if hasattr(e, 'aliases') and e.aliases:
                for alias in e.aliases.split(','):
                    alias = alias.strip()
                    if alias:
                        base_names.append(alias)
            
            smart_phrases = set()
            
            def get_locative_variants(base_name):
                variants = {base_name}
                for rule in inflection_rules:
                    if base_name.endswith(rule.suffix_in):
                        variants.add(base_name[:-len(rule.suffix_in)] + rule.suffix_out)
                return variants

            for name in base_names:
                name_lower = name.lower()
                clean_name = name_lower.replace(" do ", " ").replace(" w ", " ")
                
                name_variants = get_locative_variants(name_lower)
                clean_variants = get_locative_variants(clean_name)
                
                # Basic prefixes
                for prefix in ["", "włącz ", "wyłącz ", "ustaw ", "zgaś ", "zapal ", "otwórz ", "zamknij "]:
                    smart_phrases.add(f"{prefix}{name}")
                    smart_phrases.add(f"{prefix}{clean_name}")
                
                if e.domain in ["light", "switch"] and "światł" not in name_lower:
                    for v in clean_variants:
                        smart_phrases.add(f"światło {v}")
                        smart_phrases.add(f"włącz światło {v}")
                        smart_phrases.add(f"zapal światło {v}")
                        smart_phrases.add(f"zgaś światło {v}")
                
                elif e.domain == "cover" and "rolet" not in name_lower:
                    for v in clean_variants:
                        smart_phrases.add(f"rolety {v}")
                        smart_phrases.add(f"zasłoń rolety {v}")
                        smart_phrases.add(f"zamknij rolety {v}")
                        smart_phrases.add(f"otwórz rolety {v}")
                
                elif e.domain == "vacuum":
                    for v in clean_variants:
                        smart_phrases.add(f"odkurzacz {v}")
                        smart_phrases.add(f"odkurz {v}")
                        smart_phrases.add(f"posprzątaj {v}")
                
                elif e.entity_id.startswith("area."):
                    for v in name_variants:
                        smart_phrases.add(f"światło w {v}")
                        smart_phrases.add(f"włącz światło w {v}")
                        smart_phrases.add(f"zapal światło w {v}")
                        smart_phrases.add(f"zgaś światło w {v}")
                        smart_phrases.add(f"wszystkie światła w {v}")
                        smart_phrases.add(f"włącz wszystkie światła w {v}")
                        smart_phrases.add(f"zgaś wszystkie światła w {v}")
                        smart_phrases.add(f"rolety w {v}")
                        smart_phrases.add(f"zasłoń rolety w {v}")
                        smart_phrases.add(f"zamknij rolety w {v}")
                        smart_phrases.add(f"otwórz rolety w {v}")
                        smart_phrases.add(f"wszystkie rolety w {v}")
                        smart_phrases.add(f"odkurz {v}")
                        smart_phrases.add(f"posprzątaj w {v}")

            for phrase in smart_phrases:
                entity_phrases[e.entity_id].add(phrase.lower())

        # Deduplicate phrases
        phrase_to_eids = defaultdict(list)
        for eid, phrases in entity_phrases.items():
            for p in phrases:
                phrase_to_eids[p].append(eid)

        for eid, phrases in entity_phrases.items():
            unique_phrases = set()
            for p in phrases:
                eids = phrase_to_eids[p]
                if len(eids) == 1:
                    unique_phrases.add(p)
                else:
                    # Conflict handling: Areas win over specific entities for generic phrases
                    is_area = eid.startswith("area.")
                    has_area = any(id.startswith("area.") for id in eids)
                    
                    if not is_area and has_area:
                        # Specific entity loses to Area
                        continue
                    elif is_area:
                        # I am an area. Do I conflict with another area?
                        areas = [id for id in eids if id.startswith("area.")]
                        if len(areas) == 1:
                            unique_phrases.add(p)
                    else:
                        # Only specific entities conflicting
                        specifics = [id for id in eids if not id.startswith("area.")]
                        if len(specifics) == 1:
                            unique_phrases.add(p)

            for phrase in unique_phrases:
                ent_sentences.append(phrase)
                ent_labels.append(eid)

        X_ent = get_embeddings(ent_sentences, tokenizer, bert)

        le_entity  = LabelEncoder().fit(ent_labels)
        clf_entity = LinearSVC(C=10.0, class_weight="balanced", dual=False, max_iter=5000).fit(X_ent, le_entity.transform(ent_labels))
    else:
        logger.info("Skipping entity classifier (no data). Creating dummies.")
        # Dummy models to avoid breaking classification pipeline
        X_ent = get_embeddings(["dummy"], tokenizer, bert)
        le_entity = LabelEncoder().fit(["dummy_entity_1", "dummy_entity_2"])
        X_dummy = np.vstack([X_ent[0], X_ent[0]]) if len(X_ent) > 0 else np.zeros((2, 768))
        clf_entity = LinearSVC().fit(X_dummy, le_entity.transform(["dummy_entity_1", "dummy_entity_2"]))

    # ── Persist ───────────────────────────────────────────
    joblib.dump(clf_intent,  OUT / "intent_clf.joblib")
    joblib.dump(clf_entity,  OUT / "entity_clf.joblib")
    joblib.dump(le_intent,   OUT / "label_enc_intent.joblib")
    joblib.dump(le_entity,   OUT / "label_enc_entity.joblib")
    logger.info("Training complete. Artefacts saved to models/")

if __name__ == "__main__":
    train()
