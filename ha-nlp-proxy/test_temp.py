from nlp import load_models, classify

load_models()

text = "włącz światło w kuchni"
intent, entity, i_score, e_score = classify(text, threshold=0.0, return_all=True)

print(f"Intent: {intent} ({i_score*100:.2f}%)")
print(f"Entity: {entity} ({e_score*100:.2f}%)")

