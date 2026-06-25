from unittest.mock import patch, MagicMock
import numpy as np

# Mock the globals in nlp before importing
with patch('nlp.AutoTokenizer.from_pretrained'), patch('nlp.AutoModel.from_pretrained'), patch('joblib.load') as mock_load:
    
    mock_clf_intent = MagicMock()
    mock_clf_intent.decision_function.return_value.max.return_value = 1.0
    mock_clf_intent.predict.return_value = [0]
    
    mock_clf_entity = MagicMock()
    mock_clf_entity.decision_function.return_value.max.return_value = 1.0
    mock_clf_entity.predict.return_value = [0]
    
    mock_le_intent = MagicMock()
    mock_le_intent.inverse_transform.return_value = ["HassTurnOn"]
    
    mock_le_entity = MagicMock()
    mock_le_entity.inverse_transform.return_value = ["light.salon"]

    # Assign mocks depending on file
    def load_side_effect(path):
        name = path.name
        if "intent_clf" in name: return mock_clf_intent
        if "entity_clf" in name: return mock_clf_entity
        if "label_enc_intent" in name: return mock_le_intent
        if "label_enc_entity" in name: return mock_le_entity
    
    mock_load.side_effect = load_side_effect

    import nlp

def test_classify_no_models_loaded():
    # If classifiers are None
    nlp._clf_intent = None
    nlp._clf_entity = None
    intent, entity, iscore, escore = nlp.classify("test", 0.6)
    assert intent is None
    assert entity is None

def test_classify_high_confidence():
    # Setup mocks
    nlp._tokenizer = MagicMock()
    mock_bert_out = MagicMock()
    mock_bert_out.last_hidden_state = torch_tensor = MagicMock()
    # Mocking torch tensor operations to return dummy numpy array
    torch_tensor.__getitem__.return_value.squeeze.return_value.numpy.return_value.reshape.return_value = np.array([[1]])
    nlp._bert = MagicMock(return_value=mock_bert_out)
    
    nlp._clf_intent = mock_clf_intent
    nlp._clf_entity = mock_clf_entity
    nlp._le_intent = mock_le_intent
    nlp._le_entity = mock_le_entity
    
    mock_clf_intent.decision_function.return_value.max.return_value = 0.8
    mock_clf_entity.decision_function.return_value.max.return_value = 0.7
    
    intent, entity, iscore, escore = nlp.classify("włącz światło", 0.6)
    assert intent == "HassTurnOn"
    assert entity == "light.salon"
    assert iscore == 0.8
    assert escore == 0.7

def test_classify_low_confidence():
    mock_clf_intent.decision_function.return_value.max.return_value = 0.5 # < 0.6
    
    intent, entity, iscore, escore = nlp.classify("zrób coś dziwnego", 0.6)
    assert intent is None
    assert entity is None
    assert iscore == 0.5
