import json
from inference import ModelWrapper

# Global variable (but NOT initialized yet)
model_wrapper = None

def handler(event, context):
    global model_wrapper

    try:
        # Lazy load model (runs once per container)
        if model_wrapper is None:
            print("🔹 Loading ONNX model...")
            model_wrapper = ModelWrapper("latest.onnx", score_thr=0.15)
            print("✅ Model loaded")

        image_b64 = event.get("image_b64")
        if not image_b64:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "No image_b64 provided"})
            }

        img_b64_out, outputs = model_wrapper.predict(image_b64)

        return {
            "statusCode": 200,
            "body": json.dumps({
                "image_base64": img_b64_out
            })
        }

    except Exception as e:
        print("❌ Error:", str(e))
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
