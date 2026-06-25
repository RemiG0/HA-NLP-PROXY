import nlp
import numpy as np
import torch

nlp.load_models()
text = "zapal światło w sypialni"
inputs = nlp._tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
with torch.no_grad():
    emb = nlp._bert(**inputs).last_hidden_state[:, 0, :].squeeze().numpy().reshape(1, -1)
e_logits = nlp._clf_entity.decision_function(emb)[0]

def sigmoid_scale(margins, a=10):
    if np.isscalar(margins) or margins.ndim == 0:
        margins = np.array([margins])
    return 1.0 / (1.0 + np.exp(-margins * a))

scores = sigmoid_scale(e_logits, a=10)
top_idx = np.argsort(scores)[-5:][::-1]
print(f"Test: '{text}'")
for idx in top_idx:
    print(f"  {nlp._le_entity.inverse_transform([idx])[0]}: {scores[idx]*100:.2f}% (margin: {e_logits[idx]:.4f})")

print("---")
# Also let's check exact area phrase
text = "zapal światło w sypialnia"
inputs = nlp._tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
with torch.no_grad():
    emb = nlp._bert(**inputs).last_hidden_state[:, 0, :].squeeze().numpy().reshape(1, -1)
e_logits = nlp._clf_entity.decision_function(emb)[0]
scores = sigmoid_scale(e_logits, a=10)
top_idx = np.argsort(scores)[-5:][::-1]
print(f"Test: '{text}'")
for idx in top_idx:
    print(f"  {nlp._le_entity.inverse_transform([idx])[0]}: {scores[idx]*100:.2f}% (margin: {e_logits[idx]:.4f})")
