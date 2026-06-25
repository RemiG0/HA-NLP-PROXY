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

def get_embeddings(sentences, tokenizer, model):
    embeddings = []
    for sent in sentences:
        inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            out = model(**inputs)
        embeddings.append(out.last_hidden_state[:, 0, :].squeeze().numpy())
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
    clf_intent = LinearSVC().fit(X, le_intent.transform(intent_lbls))

    # ── Entity classifier ─────────────────────────────────
    if entities:
        # Training sentences = friendly_name
        ent_sentences = [e.friendly_name for e in entities]
        ent_labels    = [e.entity_id     for e in entities]

        X_ent = get_embeddings(ent_sentences, tokenizer, bert)

        le_entity  = LabelEncoder().fit(ent_labels)
        clf_entity = LinearSVC().fit(X_ent, le_entity.transform(ent_labels))
    else:
        logger.info("Skipping entity classifier (no data). Creating dummies.")
        # Dummy models to avoid breaking classification pipeline
        X_ent = get_embeddings(["dummy"], tokenizer, bert)
        le_entity = LabelEncoder().fit(["dummy_entity"])
        clf_entity = LinearSVC().fit(X_ent, le_entity.transform(["dummy_entity"]))

    # ── Persist ───────────────────────────────────────────
    joblib.dump(clf_intent,  OUT / "intent_clf.joblib")
    joblib.dump(clf_entity,  OUT / "entity_clf.joblib")
    joblib.dump(le_intent,   OUT / "label_enc_intent.joblib")
    joblib.dump(le_entity,   OUT / "label_enc_entity.joblib")
    logger.info("Training complete. Artefacts saved to models/")

if __name__ == "__main__":
    train()
