import base64
from inference import ModelWrapper

def test_model_loads():
    # Test if model loads without error
    model = ModelWrapper("./latest.onnx")
    assert model is not None

def test_predict_on_sample_image():
    # Read sample image and convert to base64
    with open("./test_inference/test_img.jpg", "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    # Initialize model
    model = ModelWrapper("./latest.onnx")

    # Run prediction
    img_b64_out, outputs = model.predict(img_b64)

    # Check output is a non-empty base64 string
    assert isinstance(img_b64_out, str)
    assert len(img_b64_out) > 0

    # Optional: check outputs contains expected keys/structure
    assert isinstance(outputs, tuple) or isinstance(outputs, list)
