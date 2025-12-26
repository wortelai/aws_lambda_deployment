import base64
import os
from inference import ModelWrapper

MODEL_PATH = "./latest.onnx"
TEST_IMAGE_PATH = "./test_inference/test_img.jpg"

def test_model_loads():
    assert os.path.exists(MODEL_PATH), "ONNX model file not found"
    model = ModelWrapper(MODEL_PATH)
    assert model is not None


def test_predict_on_sample_image():
    assert os.path.exists(TEST_IMAGE_PATH), "Test image not found"

    # Read sample image and convert to base64
    with open(TEST_IMAGE_PATH, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    model = ModelWrapper(MODEL_PATH)

    img_b64_out, outputs = model.predict(img_b64)

    # Output image checks
    assert isinstance(img_b64_out, str)
    assert len(img_b64_out) > 0

    # Detection output structure
    assert isinstance(outputs, (tuple, list))
