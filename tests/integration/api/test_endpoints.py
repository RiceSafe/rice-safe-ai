import sys
import os
import io
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from PIL import Image
import numpy as np

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Import api inside a patch block or after mocking imports if necessary
# But api imports pytorch/tensorflow at top level, which takes time. 
# For unit/integration tests, we might want to mock the heavy imports if we just want to test logic.
# However, to test FastAPI app integration, we usually import the app.
# We will rely on mocking the startup event or the globals.

from api import app, PREDICTION_MAP

client = TestClient(app)

# Helper to create dummy image bytes
def create_dummy_image():
    img = Image.new('RGB', (224, 224), color='green')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    return img_byte_arr.getvalue()

@pytest.fixture
def mock_api_dependencies():
    """
    Mock all global models in api to avoid loading heavy models during tests.
    """
    with patch('api.keras_model') as mock_keras, \
         patch('api.label_encoder') as mock_le, \
         patch('api.scaler') as mock_scaler, \
         patch('api.text_tokenizer') as mock_tok, \
         patch('api.text_model_embedder') as mock_embed, \
         patch('api.mobilenet_extractor') as mock_mobilenet, \
         patch('api.clip_model') as mock_clip, \
         patch('api.clip_processor') as mock_clip_proc, \
         patch('api.check_rice_leaf_with_clip') as mock_check_clip:
        
        # Setup common mock behaviors
        
        # Mock Label Encoder
        # Use Thai labels as per modelV2.py
        mock_le.classes_ = ['ปกติ', 'โรคขอบใบแห้ง', 'โรคใบขีดโปร่งแสง', 'โรคใบจุดสีน้ำตาล', 'โรคไหม้']
        mock_le.inverse_transform.side_effect = lambda x: [mock_le.classes_[i] for i in x]
        
        # Mock Scaler
        mock_scaler.transform.return_value = np.zeros((1, 2304))
        
        # Mock MobileNet
        mock_mobilenet.predict.return_value = np.zeros((1, 1280))
        
        # Mock Text Embedder (PyTorch output)
        mock_embed_output = MagicMock()
        mock_embed_output.last_hidden_state.mean.return_value.cpu.return_value.numpy.return_value.squeeze.return_value = np.zeros(1024)
        mock_embed.return_value = mock_embed_output
        
        yield {
            'keras': mock_keras,
            'le': mock_le,
            'scaler': mock_scaler,
            'check_clip': mock_check_clip
        }

def test_health_check_healthy(mock_api_dependencies):
    # Tests rely on models being "loaded" (which our mocks simulate by being present in globals)
    # The health check code checks: all([keras_model, ...])
    # Since we patched them in api namespace, they should be truthy mock objects.
    
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()['status'] == "healthy"

def test_predict_normal_case(mock_api_dependencies):
    mocks = mock_api_dependencies
    
    # 1. Mock CLIP check to PASS
    mocks['check_clip'].return_value = {
        "is_rice_leaf": True,
        "confidence": 0.95,
        "matched_label": "rice leaf",
        "all_scores": {}
    }
    
    # 2. Mock Keras Prediction (High Confidence Blast = โรคไหม้)
    # Index 4 = โรคไหม้
    mock_probs = np.array([[0.02, 0.02, 0.01, 0.05, 0.90]]) 
    
    # Keras model call returns tensor, convert to numpy
    mock_tensor = MagicMock()
    mock_tensor.numpy.return_value = mock_probs
    mocks['keras'].return_value = mock_tensor
    
    # Send Request
    files = {'image': ('test.jpg', create_dummy_image(), 'image/jpeg')}
    data = {'description': 'Found spots on leaves'}
    
    response = client.post("/predict/", files=files, data=data)
    
    assert response.status_code == 200
    res_json = response.json()
    
    assert res_json['prediction'] == 'rice_blast' # Should match PREDICTION_MAP key/value
    assert res_json['clip_check']['passed'] == True
    # Confidence should be approx 90.00%
    assert "90.00" in res_json['confidence']

def test_predict_not_rice_leaf(mock_api_dependencies):
    mocks = mock_api_dependencies
    
    # Mock CLIP check to FAIL
    mocks['check_clip'].return_value = {
        "is_rice_leaf": False,
        "confidence": 0.10,
        "matched_label": "cats",
        "all_scores": {}
    }
    
    files = {'image': ('cat.jpg', create_dummy_image(), 'image/jpeg')}
    data = {'description': 'My cat'}
    
    response = client.post("/predict/", files=files, data=data)
    
    assert response.status_code == 200
    res_json = response.json()
    
    assert res_json['prediction'] == 'not_rice'
    assert res_json['clip_check']['passed'] == False

def test_predict_low_confidence(mock_api_dependencies):
    mocks = mock_api_dependencies
    
    # Mock CLIP PASS
    mocks['check_clip'].return_value = {
        "is_rice_leaf": True,
        "confidence": 0.90,
        "matched_label": "rice leaf",
        "all_scores": {}
    }
    
    # Mock Keras Prediction (Low Confidence, e.g. < 80%)
    # Let's say 40% only
    mock_probs = np.array([[0.40, 0.20, 0.20, 0.10, 0.10]])
    
    mock_tensor = MagicMock()
    mock_tensor.numpy.return_value = mock_probs
    mocks['keras'].return_value = mock_tensor
    
    files = {'image': ('test.jpg', create_dummy_image(), 'image/jpeg')}
    data = {'description': 'Some text'}
    
    response = client.post("/predict/", files=files, data=data)
    
    assert response.status_code == 200
    res_json = response.json()
    
    assert res_json['prediction'] == 'not_clear'
    assert "40.00" in res_json['confidence']
