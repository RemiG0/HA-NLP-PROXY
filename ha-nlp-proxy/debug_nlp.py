from nlp import load_models, classify
load_models()
i, e, i_s, e_s = classify("włącz światło w kuchni", 0.0, True)
print(f"Intent: {i} {i_s}")
print(f"Entity: {e} {e_s}")
