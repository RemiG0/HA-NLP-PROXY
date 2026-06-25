import db
import nlp
import train
import numpy as np
from sqlmodel import Session, select
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import torch

nlp.load_models()

with Session(db.engine) as s:
    entities = s.exec(select(db.Entity).where(db.Entity.enabled == True)).all()

ent_sentences = []
ent_labels = []
entity_phrases = defaultdict(set)

for e in entities:
    base_names = [e.friendly_name]
    if hasattr(e, 'aliases') and e.aliases:
        for alias in e.aliases.split(','):
            alias = alias.strip()
            if alias: base_names.append(alias)
    
    smart_phrases = set()
    for name in base_names:
        name_lower = name.lower()
        clean_name = name_lower.replace(" do ", " ").replace(" w ", " ")
        name_variants = train.get_locative_variants(name_lower)
        clean_variants = train.get_locative_variants(clean_name)
        
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
                smart_phrases.add(f"zamknij rolety {v}")
        elif e.domain == "vacuum":
            for v in clean_variants:
                smart_phrases.add(f"odkurz {v}")
        elif e.entity_id.startswith("area."):
            for v in name_variants:
                smart_phrases.add(f"włącz światło w {v}")
                smart_phrases.add(f"wszystkie światła w {v}")

    entity_phrases[e.entity_id].update(smart_phrases)

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
            is_area = eid.startswith("area.")
            has_specific = any(not id.startswith("area.") for id in eids)
            if is_area and has_specific: continue
            unique_phrases.add(p)

    for phrase in unique_phrases:
        ent_sentences.append(phrase)
        ent_labels.append(eid)

# Compute embeddings
X_ent = train.get_embeddings(ent_sentences, nlp._tokenizer, nlp._bert)

def test_query(text):
    inputs = nlp._tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        emb = nlp._bert(**inputs).last_hidden_state[:, 0, :].squeeze().numpy().reshape(1, -1)
    
    sims = cosine_similarity(emb, X_ent)[0]
    best_idx = np.argmax(sims)
    print(f"Test: '{text}' -> {ent_labels[best_idx]} (score: {sims[best_idx]*100:.1f}%) [Matched phrase: {ent_sentences[best_idx]}]")

test_query("włącz wszystkie światła w kuchni")
test_query("włącz światło w kuchni")
test_query("włącz wiatrak")
test_query("zapal światło w sypialni")
