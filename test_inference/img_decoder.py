import json
import base64
import os
def save_base64_image_from_nested_json(json_path, output_image_name):
    # Step 1: load outer JSON
    with open(json_path, "r") as f:
        outer_data = json.load(f)

    # Step 2: parse inner JSON from 'body' string
    inner_json_str = outer_data.get("body")
    if not inner_json_str:
        raise ValueError("No 'body' key found in JSON")

    inner_data = json.loads(inner_json_str)

    # Step 3: get base64 string
    img_b64 = inner_data.get("image_base64")
    if not img_b64:
        raise ValueError("No 'image_base64' key found inside 'body'")

    # Step 4: decode and save image
    img_bytes = base64.b64decode(img_b64)
    with open(output_image_name, "wb") as img_file:
        img_file.write(img_bytes)

    print(f"Image saved as {os.path.abspath(output_image_name)}")

# Example usage:
save_base64_image_from_nested_json("/opt/workspace_daniyal/ML_model_deploy/aws_lambda/output_base64.json", "output_image1.jpg")
