import nlp
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

nlp.load_models()

def evaluate(text):
    inputs = nlp._tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        emb = nlp._bert(**inputs).last_hidden_state[:, 0, :].squeeze().numpy().reshape(1, -1)
    
    # We don't have X_ent saved in nlp.py. We need to save X_ent during training to use cosine similarity.
    # Since LinearSVC is saved, its weights (clf.coef_) are actually the normal vectors for each class!
    # Cosine similarity to the SVM class weights?
    weights = nlp._clf_entity.coef_
    sims = cosine_similarity(emb, weights)[0]
    
    top_idx = np.argsort(sims)[-5:][::-1]
    print(f"\nTest: '{text}'")
    for idx in top_idx:
        print(f"  {nlp._le_entity.inverse_transform([idx])[0]}: {sims[idx]*100:.2f}%")

evaluate("włącz wszystkie światła w kuchni")
evaluate("zapal światło w sypialni")
evaluate("zamknij rolety w salonie")
evaluate("włącz wiatrak")
