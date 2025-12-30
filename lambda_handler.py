import json
from inference import ModelWrapper

model_wrapper = None

def handler(event, context):
    global model_wrapper

    try:
        if model_wrapper is None:
            print("🔹 Loading ONNX model...")
            model_wrapper = ModelWrapper("latest.onnx", score_thr=0.15)
            print("✅ Model loaded")

        # ✅ Parse body correctly
        body = event.get("body")
        if body is None:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Missing request body"})
            }

        if isinstance(body, str):
            body = json.loads(body)

        image_b64 = body.get("image_b64")
        if not image_b64:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "No image_b64 provided"})
            }

        img_b64_out, outputs = model_wrapper.predict(image_b64)

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
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
