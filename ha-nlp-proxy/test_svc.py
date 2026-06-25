import nlp
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from db import engine, Entity
from sqlmodel import Session, select
import torch
import numpy as np

nlp.load_models()
PREFIXES = ["", "włącz ", "wyłącz ", "ustaw ", "zgaś ", "zapal ", "otwórz ", "zamknij "]

with Session(engine) as session:
    entities = session.exec(select(Entity).where(Entity.enabled == True)).all()

ent_sentences = []
ent_labels = []
for e in entities:
    base_names = [e.friendly_name]
    if hasattr(e, 'aliases') and e.aliases:
        for alias in e.aliases.split(','):
            alias = alias.strip()
            if alias:
                base_names.append(alias)
    
    for name in base_names:
        for prefix in PREFIXES:
            ent_sentences.append(f"{prefix}{name}".lower())
            ent_labels.append(e.entity_id)

def get_embeddings(sentences):
    embeddings = []
    for i in range(0, len(sentences), 32):
        batch = sentences[i:i+32]
        inputs = nlp._tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            emb = nlp._bert(**inputs).last_hidden_state[:, 0, :].numpy()
        embeddings.append(emb)
    return np.vstack(embeddings)

X_ent = get_embeddings(ent_sentences)
le_entity = LabelEncoder().fit(ent_labels)
clf_entity = LinearSVC(C=1.0, class_weight="balanced", dual=False, max_iter=5000).fit(X_ent, le_entity.transform(ent_labels))

def test_sent(text):
    inputs = nlp._tokenizer(text.lower(), return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        emb = nlp._bert(**inputs).last_hidden_state[:, 0, :].squeeze().numpy().reshape(1, -1)
    
    margins = clf_entity.decision_function(emb)[0]
    top_idx = np.argsort(margins)[-3:][::-1]
    
    print(f"\nTest: '{text}'")
    for idx in top_idx:
        print(f"  {le_entity.inverse_transform([idx])[0]}: {margins[idx]:.4f}")

test_sent("włącz światło w kuchni")
test_sent("zapal światło w sypialni")
test_sent("zamknij rolety")
