import sys
import os
import io
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from PIL import Image
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from api import app

client = TestClient(app)


def create_dummy_image():
    img = Image.new("RGB", (224, 224), color="green")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="JPEG")
    return img_byte_arr.getvalue()


def _ensemble_probs(n_classes: int, peak_idx: int, peak: float = 1.0):
    p = np.zeros((1, n_classes), dtype=np.float32)
    p[0, peak_idx] = peak
    if peak < 1.0:
        rest = (1.0 - peak) / max(n_classes - 1, 1)
        for i in range(n_classes):
            if i != peak_idx:
                p[0, i] = rest
    return p


@pytest.fixture
def mock_api_dependencies():
    """Mock globals in api so tests skip heavy model load behavior."""
    with patch("api.keras_model") as mock_keras, \
         patch("api.label_encoder") as mock_le, \
         patch("api.bge_tokenizer") as mock_bge_tok, \
         patch("api.bge_model") as mock_bge, \
         patch("api.clip_model") as mock_clip, \
         patch("api.clip_processor") as mock_clip_proc, \
         patch("api.check_rice_leaf_with_clip") as mock_check_clip:

        mock_le.classes_ = [
            "ปกติ",
            "โรคขอบใบแห้ง",
            "โรคใบขีดโปร่งแสง",
            "โรคใบจุดสีน้ำตาล",
            "โรคไหม้",
        ]
        mock_le.inverse_transform.side_effect = lambda x: [mock_le.classes_[i] for i in x]

        yield {
            "keras": mock_keras,
            "le": mock_le,
            "check_clip": mock_check_clip,
        }


def test_health_check_healthy(mock_api_dependencies):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_normal_case(mock_api_dependencies):
    mocks = mock_api_dependencies

    mocks["check_clip"].return_value = {
        "is_rice_leaf": True,
        "confidence": 0.95,
        "matched_label": "a photo of a rice leaf or rice plant",
        "all_scores": {},
    }

    blast_idx = 4
    p = _ensemble_probs(5, blast_idx, peak=1.0)
    mocks["keras"].predict.return_value = [p, p.copy(), p.copy()]

    files = {"image": ("test.jpg", create_dummy_image(), "image/jpeg")}
    data = {"description": "Found spots on leaves"}

    response = client.post("/predict/", files=files, data=data)

    assert response.status_code == 200
    res_json = response.json()

    assert res_json["prediction"] == "rice_blast"
    assert res_json["clip_check"]["passed"] is True
    assert "100.00" in res_json["confidence"]


def test_predict_not_rice_leaf(mock_api_dependencies):
    mocks = mock_api_dependencies

    mocks["check_clip"].return_value = {
        "is_rice_leaf": False,
        "confidence": 0.10,
        "matched_label": "cats",
        "all_scores": {},
    }

    files = {"image": ("cat.jpg", create_dummy_image(), "image/jpeg")}
    data = {"description": "My cat"}

    response = client.post("/predict/", files=files, data=data)

    assert response.status_code == 200
    res_json = response.json()

    assert res_json["prediction"] == "not_rice"
    assert res_json["clip_check"]["passed"] is False


def test_predict_low_confidence(mock_api_dependencies):
    mocks = mock_api_dependencies

    mocks["check_clip"].return_value = {
        "is_rice_leaf": True,
        "confidence": 0.90,
        "matched_label": "a photo of a rice leaf or rice plant",
        "all_scores": {},
    }

    p = np.array([[0.40, 0.20, 0.20, 0.10, 0.10]], dtype=np.float32)
    mocks["keras"].predict.return_value = [p, p.copy(), p.copy()]

    files = {"image": ("test.jpg", create_dummy_image(), "image/jpeg")}
    data = {"description": "Some text"}

    response = client.post("/predict/", files=files, data=data)

    assert response.status_code == 200
    res_json = response.json()

    assert res_json["prediction"] == "not_clear"
    assert "40.00" in res_json["confidence"]
