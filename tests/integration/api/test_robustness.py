import sys
import os
import io
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from api import app

client = TestClient(app)


@pytest.fixture
def mock_api_deps():
    with patch("api.keras_model"), \
         patch("api.label_encoder"), \
         patch("api.bge_tokenizer"), \
         patch("api.bge_model"), \
         patch("api.clip_model"), \
         patch("api.clip_processor"), \
         patch("api.check_rice_leaf_with_clip") as mock_clip:

        mock_clip.return_value = {
            "is_rice_leaf": True,
            "confidence": 0.99,
            "matched_label": "rice",
            "all_scores": {},
        }
        yield


def test_api_invalid_file_type(mock_api_deps):
    files = {"image": ("test.txt", b"This is not an image", "text/plain")}
    data = {"description": "test"}

    response = client.post("/predict/", files=files, data=data)

    assert response.status_code in [400, 500]
    detail = response.json().get("detail", "")
    assert "failed" in str(detail).lower() or "error" in str(detail).lower()


def test_api_missing_description(mock_api_deps):
    img = Image.new("RGB", (8, 8), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    files = {"image": ("test.jpg", buf.getvalue(), "image/jpeg")}

    response = client.post("/predict/", files=files)

    assert response.status_code == 422
