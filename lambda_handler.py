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

        # Handle both Lambda Test UI and Function URL
        if "body" in event:
            body = event["body"]
            if isinstance(body, str):
                body = json.loads(body)
        else:
            body = event

        image_b64 = body.get("image_b64")
        if not image_b64:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "No image_b64 provided"})
            }

        detections = model_wrapper.predict(image_b64)

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "detections": detections
            })
        }

    except Exception as e:
        print("❌ Error:", str(e))
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
