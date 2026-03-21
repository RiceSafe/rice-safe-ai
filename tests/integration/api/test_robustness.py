import sys
import os
import io
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from api import app

client = TestClient(app)

@pytest.fixture
def mock_api_deps():
    # Minimal mocks to pass startup checks and get to endpoint logic
    with patch('api.keras_model'), \
         patch('api.label_encoder'), \
         patch('api.scaler'), \
         patch('api.text_tokenizer'), \
         patch('api.text_model_embedder'), \
         patch('api.mobilenet_extractor'), \
         patch('api.clip_model'), \
         patch('api.clip_processor'), \
         patch('api.check_rice_leaf_with_clip') as mock_clip:
         
        mock_clip.return_value = {"is_rice_leaf": True, "confidence": 0.99, "matched_label": "rice"}
        yield

def test_api_invalid_file_type(mock_api_deps):
    """Test sending a text file instead of an image."""
    files = {'image': ('test.txt', b'This is not an image', 'text/plain')}
    data = {'description': 'test'}
    
    # API usually requires UploadFile. 
    # PIL.Image.open will fail if content is not image.
    # api catches Exception and raises 500 or 400.
    # In predict_endpoint:
    # pil_image = Image.open(...) -> invalid -> UnidentifiedImageError
    # The try/except block catches generic Exception -> 500.
    
    response = client.post("/predict/", files=files, data=data)
    
    # Ideally should be 400 or 500 (handled error), not app crash
    assert response.status_code in [400, 500]
    assert "failed" in response.json()['detail'].lower() or "error" in response.json()['detail'].lower()

def test_api_missing_description(mock_api_deps):
    """Test sending request without description."""
    files = {'image': ('test.jpg', b'\xFF\xD8\xFF', 'image/jpeg')} # Minimal dummy jpg header
    
    response = client.post("/predict/", files=files) 
    
    # 422 Validation Error (FastAPI default for missing form field)
    assert response.status_code == 422
