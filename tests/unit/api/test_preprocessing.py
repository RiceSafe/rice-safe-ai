import sys
import os
import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from PIL import Image
from fastapi import HTTPException

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from api import preprocess_input_for_model


@pytest.fixture
def mock_preprocess_deps():
    with patch("api.keras_model") as mock_keras, \
         patch("api.bge_tokenizer") as mock_tok, \
         patch("api.bge_model") as mock_bge:

        mock_batch = MagicMock()
        mock_batch.to.return_value = mock_batch
        mock_tok.return_value = mock_batch

        mock_out = MagicMock()
        emb = np.zeros((1, 1024), dtype=np.float32)
        mock_out.last_hidden_state.mean.return_value.cpu.return_value.numpy.return_value = emb
        mock_bge.return_value = mock_out

        yield {
            "tokenizer": mock_tok,
            "bge": mock_bge,
        }


def test_preprocess_input_for_model_success(mock_preprocess_deps):
    img = Image.new("RGB", (224, 224), color="red")
    description = "Test description"

    result = preprocess_input_for_model(img, description)

    assert "image_input" in result and "text_input" in result
    assert result["image_input"].shape == (1, 224, 224, 3)
    assert result["text_input"].shape == (1, 1024)
    mock_preprocess_deps["tokenizer"].assert_called_once()
    mock_preprocess_deps["bge"].assert_called_once()


def test_preprocess_input_for_model_not_loaded(mock_preprocess_deps):
    with patch("api.keras_model", None):
        with pytest.raises(HTTPException) as excinfo:
            preprocess_input_for_model(Image.new("RGB", (10, 10)), "test")
        assert excinfo.value.status_code == 503
        assert "Models not fully loaded" in excinfo.value.detail
