from nlp import load_models, classify

load_models()

text = "włącz światło w kuchni"
intent, entity, i_score, e_score = classify(text, threshold=0.0, return_all=True)

print(f"Intent: {intent} ({i_score*100:.2f}%)")
print(f"Entity: {entity} ({e_score*100:.2f}%)")

from nlp import _tokenizer, _bert, _clf_entity, _le_entity
import torch
import numpy as np

inputs = _tokenizer(text.lower(), return_tensors="pt", truncation=True, max_length=128)
with torch.no_grad():
    emb = _bert(**inputs).last_hidden_state[:, 0, :].squeeze().numpy().reshape(1, -1)

probas = _clf_entity.predict_proba(emb)[0]
top5_idx = np.argsort(probas)[-5:][::-1]

print("\nTop 5 Entity Classes:")
for idx in top5_idx:
    class_name = _le_entity.inverse_transform([idx])[0]
    prob = probas[idx]
    print(f"{class_name}: {prob*100:.2f}%")
