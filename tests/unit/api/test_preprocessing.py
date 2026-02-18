import sys
import os
import io
import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from PIL import Image

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from api import preprocess_input_for_model

@pytest.fixture
def mock_preprocess_deps():
    with patch('api.keras_model') as mock_keras, \
         patch('api.label_encoder') as mock_le, \
         patch('api.scaler') as mock_scaler, \
         patch('api.text_tokenizer') as mock_tok, \
         patch('api.text_model_embedder') as mock_embed, \
         patch('api.mobilenet_extractor') as mock_mobilenet:
        
        # Setup mocks
        mock_scaler.transform.return_value = np.zeros((1, 2304))
        mock_mobilenet.predict.return_value = np.zeros((1, 1280))
        
        # Mock Text Embedder output
        mock_embed_output = MagicMock()
        mock_embed_output.last_hidden_state.mean.return_value.cpu.return_value.numpy.return_value.squeeze.return_value = np.zeros(1024)
        mock_embed.return_value = mock_embed_output
        
        yield {
            'scaler': mock_scaler,
            'mobilenet': mock_mobilenet,
            'tokenizer': mock_tok
        }

def test_preprocess_input_for_model_success(mock_preprocess_deps):
    # Dummy image bytes
    img = Image.new('RGB', (224, 224), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    image_bytes = img_byte_arr.getvalue()
    
    description = "Test description"
    
    result = preprocess_input_for_model(image_bytes, description)
    
    assert result.shape == (1, 2304)
    # Check if dependencies were called
    mock_preprocess_deps['mobilenet'].predict.assert_called_once()
    mock_preprocess_deps['tokenizer'].assert_called_once()
    mock_preprocess_deps['scaler'].transform.assert_called_once()

def test_preprocess_input_for_model_not_loaded(mock_preprocess_deps):
    # Simulate models not loaded by setting one to None
    with patch('api.keras_model', None):
        with pytest.raises(Exception) as excinfo:
            preprocess_input_for_model(b'', "test")
        assert "Models not fully loaded" in str(excinfo.value)
