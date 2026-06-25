import joblib, torch, numpy as np, pathlib
from transformers import AutoTokenizer, AutoModel
import logging

logger = logging.getLogger("ha-nlp-proxy")

MODEL_NAME = "allegro/herbert-base-cased"
_tokenizer = None
_bert = None
_clf_intent = None
_clf_entity = None
_le_intent = None
_le_entity = None

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

def classify(text: str, threshold: float):
    if _clf_intent is None or _clf_entity is None:
        logger.warning("Classifiers not loaded, falling back immediately.")
        return None, None, 0.0, 0.0

    inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        emb = _bert(**inputs).last_hidden_state[:, 0, :].squeeze().numpy().reshape(1, -1)

    intent_score = float(_clf_intent.decision_function(emb).max())
    entity_score = float(_clf_entity.decision_function(emb).max())

    if intent_score < threshold or entity_score < threshold:
        return None, None, intent_score, entity_score

    intent    = _le_intent.inverse_transform(_clf_intent.predict(emb))[0]
    entity_id = _le_entity.inverse_transform(_clf_entity.predict(emb))[0]
    return intent, entity_id, intent_score, entity_score
