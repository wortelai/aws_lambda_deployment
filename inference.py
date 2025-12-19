import onnxruntime as ort
import cv2
import numpy as np
import base64
from onnx_yolox_model import Model

# Load model once globally for performance
AVAILABLE_PROVIDERS = ort.get_available_providers()
if "CUDAExecutionProvider" in AVAILABLE_PROVIDERS:
    DEVICE = "cuda"
    PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
else:
    DEVICE = "cpu"
    PROVIDERS = ["CPUExecutionProvider"]

class ModelWrapper:
    def __init__(self, model_path, score_thr=0.15):
        self.model = Model(model_path, conf=score_thr, nms=0.3, tsize=640, overlap=30)
        self.score_thr = score_thr

    def predict(self, img_b64):
        # Decode base64 image to numpy array
        img_data = base64.b64decode(img_b64)
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Save temp image (Lambda has /tmp writable dir)
        temp_img_path = "/tmp/input.jpg"
        cv2.imwrite(temp_img_path, img)

        # Run model inference
        outputs_list = self.model([{"image": temp_img_path}], False, False)
        outputs = outputs_list[0]

        # Draw boxes (implement your own or adapt from Flask app)
        img_with_boxes = self.draw_boxes_on_image(img, outputs)

        # Encode image back to base64
        _, buf = cv2.imencode(".jpg", img_with_boxes)
        img_b64_out = base64.b64encode(buf).decode("utf-8")

        return img_b64_out, outputs

    def draw_boxes_on_image(self, img, outputs):
        bboxes, cls_ids, scores, cls_names = outputs
        for (x1, y1, x2, y2), cid, score, name in zip(bboxes, cls_ids, scores, cls_names):
            if float(score) < self.score_thr:
                continue
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            cv2.putText(img, f"{name} {score:.2f}", (int(x1), int(y1)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        return img
