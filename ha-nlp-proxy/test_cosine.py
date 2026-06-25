import nlp
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
            ent_sentences.append(f"{prefix}{name}")
            ent_labels.append(e.entity_id)

ent_sentences = [s.lower() for s in ent_sentences]

def get_emb(text):
    inputs = nlp._tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        return nlp._bert(**inputs).last_hidden_state[:, 0, :].squeeze().numpy()

query_emb = get_emb("włącz światło w kuchni".lower())

sims = []
for i, sent in enumerate(ent_sentences):
    if "kuchni" in sent or "light" in ent_labels[i]:
        sent_emb = get_emb(sent)
        cos_sim = np.dot(query_emb, sent_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(sent_emb))
        sims.append((cos_sim, ent_labels[i], sent))

sims.sort(key=lambda x: x[0], reverse=True)
print("Top 5 Cosine Similarities:")
for s in sims[:5]:
    print(f"{s[0]:.4f} - {s[1]} (from '{s[2]}')")

