import nlp
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from db import engine, Entity
from sqlmodel import Session, select
import torch
import numpy as np

nlp.load_models()

def sigmoid_scale(margins, a=10):
    # Ensure margins is an array
    if np.isscalar(margins) or margins.ndim == 0:
        margins = np.array([margins])
    return 1.0 / (1.0 + np.exp(-margins * a))

def test_sent(text):
    inputs = nlp._tokenizer(text.lower(), return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        emb = nlp._bert(**inputs).last_hidden_state[:, 0, :].squeeze().numpy().reshape(1, -1)
    
    margins = nlp._clf_entity.decision_function(emb)[0]
    scores = sigmoid_scale(margins, a=10)
    top_idx = np.argsort(scores)[-3:][::-1]
    
    print(f"\nTest: '{text}'")
    for idx in top_idx:
        print(f"  {nlp._le_entity.inverse_transform([idx])[0]}: {scores[idx]*100:.2f}% (margin: {margins[idx]:.4f})")

test_sent("włącz światło w kuchni")
test_sent("zapal światło w sypialni")
test_sent("zamknij rolety")
