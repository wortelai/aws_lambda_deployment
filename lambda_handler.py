import json
from inference import ModelWrapper

# Load model once globally (for cold start optimization)
model_wrapper = ModelWrapper("./latest.onnx", score_thr=0.15)

def handler(event, context):
    try:
        # Lambda event will contain base64 image as JSON string
        image_b64 = event.get("image_b64")
        if not image_b64:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "No image_b64 provided"})
            }

        img_b64_out, outputs = model_wrapper.predict(image_b64)

        response = {
            "statusCode": 200,
            "body": json.dumps({
                "image_base64": img_b64_out,
                # You can add detections here as well if you adapt outputs_to_json
            })
        }
        return response

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
