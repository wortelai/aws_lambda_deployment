import base64
import json
import cv2
import requests

# ---------------- CONFIG ----------------
LAMBDA_URL = "http://localhost:9000/2015-03-31/functions/function/invocations"
INPUT_IMAGE = "/opt/workspace_daniyal/ML_model_deploy/aws_lambda/aws_lambda_deployment/test_git_action/test_img.jpg"
OUTPUT_IMAGE = "output_with_boxes.jpg"
# ----------------------------------------


def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def draw_boxes(image, detections):
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        label = f'{det["class_name"]} {det["score"]:.2f}'

        # Draw bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label background
        (w, h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(image, (x1, y1 - h - 6), (x1 + w, y1), (0, 255, 0), -1)

        # Draw label text
        cv2.putText(
            image,
            label,
            (x1, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return image


def main():
    # Load image
    image = cv2.imread(INPUT_IMAGE)
    if image is None:
        raise RuntimeError("Failed to load input image")

    # Encode image
    image_b64 = encode_image(INPUT_IMAGE)

    # Send request
    response = requests.post(
        LAMBDA_URL,
        json={"image_b64": image_b64},
        timeout=120,
    )

    if response.status_code != 200:
        raise RuntimeError(f"Lambda error: {response.text}")

    result = response.json()
    detections = json.loads(result["body"])["detections"]

    # Draw boxes
    image = draw_boxes(image, detections)

    # Save result
    cv2.imwrite(OUTPUT_IMAGE, image)
    print(f"✅ Saved result to {OUTPUT_IMAGE}")


if __name__ == "__main__":
    main()
