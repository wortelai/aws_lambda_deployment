import base64
import os
from unittest.mock import patch
from inference import ModelWrapper

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "latest.onnx"))
TEST_IMAGE_PATH = os.path.join(BASE_DIR, "test_img.jpg")

@patch("onnx_yolox_model.onnxruntime.InferenceSession")
def test_model_loads(mock_session):
    assert os.path.exists(MODEL_PATH), "ONNX model file not found"
    model = ModelWrapper(MODEL_PATH)
    assert model is not None

@patch("onnx_yolox_model.onnxruntime.InferenceSession")
def test_predict_on_sample_image(mock_session):
    assert os.path.exists(TEST_IMAGE_PATH), "Test image not found"

    with open(TEST_IMAGE_PATH, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    model = ModelWrapper(MODEL_PATH)
    outputs = model.predict(img_b64)

    assert isinstance(outputs, list)
    assert all(isinstance(d, dict) for d in outputs)
