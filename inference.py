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
        img_data = base64.b64decode(img_b64)
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        temp_img_path = "/tmp/input.jpg"
        cv2.imwrite(temp_img_path, img)

        outputs_list = self.model([{"image": temp_img_path}], False, False)
        bboxes, cls_ids, scores, cls_names = outputs_list[0]

    # Filter by score_thr and convert to plain python types for JSON
        results = []
        for (x1, y1, x2, y2), cid, score, name in zip(bboxes, cls_ids, scores, cls_names):
            score_f = float(score)
            if score_f < self.score_thr:
                continue
            results.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "class_id": int(cid),
                "class_name": str(name),
                "score": score_f
            })

        return results
