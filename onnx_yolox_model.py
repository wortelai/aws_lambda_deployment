import cv2
import numpy as np
import onnxruntime
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

class Model:
    def __init__(self, model_path, **kwargs):
        
        self.conf = kwargs["conf"]
        self.nms = kwargs["nms"]
        # Create inference session for the model
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = onnxruntime.InferenceSession(model_path, providers=providers)        
        # Get classes from onnxruntime model
        meta = self.session.get_modelmeta()
        class_info = meta.custom_metadata_map.items()
        self.class_names = list(dict(sorted(class_info)).values()) # sort classes to ensure correct order       
    
    def prepare_image(self, image_path):
        img = cv2.imread(image_path)
        r,c = img.shape[:2]
        r = -(r%32) if r%32 else None
        c = -(c%32) if c%32 else None
        img = img[: r, : c]
        return img


    def preprocess_image(self, img, h_ceil):
        if isinstance(img, str):
            raw_img = self.prepare_image(img)
        else:
            raw_img = img
        processed_img = np.expand_dims(np.moveaxis(raw_img, -1, 0), axis=0).astype(np.float32)

        if 'fp16' in self.session._model_path:
            processed_img = processed_img.astype(np.float16)
            
        h, w = processed_img.shape[2:]
        h_ceil = int(h_ceil*h) if h_ceil else h
        blob = processed_img[:,:,:h_ceil,:]
        return blob
    

    @staticmethod
    def validate_image_shapes(processed_images):
        # Extract the shapes of all images (ignoring batch and channel dimensions)
        shapes = [image.shape[2:] for image in processed_images]  # Get (H, W) for each image
        
        # Check if all shapes are the same
        if len(set(shapes)) > 1:
            raise ValueError("Images in batch do not have the same shape, either use batch size 1 or use images of same shape")


    def __call__(self, images_details, use_all_detector_classes, multi_class_nms, classification_model=None, width=None, height=None, crop_from_top=0):         
        outputs = []
        with ThreadPoolExecutor(max_workers=1) as executor:
            current_future = executor.submit(self.preprocess_image, images_details[0]['image'], crop_from_top)
            current_blob = current_future.result()

            for next_path in tqdm(images_details[1:]):
                next_future = executor.submit(self.preprocess_image, next_path['image'], crop_from_top)
                try:
                    bboxes, labels = self.session.run(None, {'input': current_blob})
                except:
                    print("An Exception occurred, check ONNX model")
                    outputs.append(([], [], [], []))
                    current_blob = next_future.result()
                    continue
                outputs.append(self.process_model_output(bboxes, labels, multi_class_nms, use_all_detector_classes))
                current_blob = next_future.result()
            # Process final image
            try:
                bboxes, labels = self.session.run(None, {'input': current_blob})
            except:
                print("An Exception occurred, check ONNX model")
                outputs.append(([], [], [], []))
            else:
                outputs.append(self.process_model_output(bboxes, labels, multi_class_nms, use_all_detector_classes))

        return outputs


    def process_model_output(self, bboxes, labels, multi_class_nms, use_all_detector_classes):
        # Below code performs nms per class  
        def get_nms_boxes_labels(bboxes, labels, multi_class_nms):
            nms_bboxes = np.empty((0, 5), dtype=np.float32)     
            nms_labels = np.empty((0), dtype=np.int8)  
            mask = (labels!=-1)
            bboxes, labels = bboxes[mask], labels[mask]
            xywh_boxes = bboxes[:, :4].copy()   # np.zeros throws error here due to high dimension of array
            scores = bboxes[:,4]
            xywh_boxes[:,2] = bboxes[:,2]-bboxes[:,0]
            xywh_boxes[:,3] = bboxes[:,3]-bboxes[:,1]  
            if multi_class_nms:
                indices = cv2.dnn.NMSBoxes(xywh_boxes, scores, score_threshold=self.conf, nms_threshold=self.nms)
                if len(indices):
                    nms_bboxes, nms_labels = bboxes[indices], labels[indices]
            else:
                for label in set(labels): 
                    mask = labels==label  
                    boxes_per_class = bboxes[mask]
                    xywh_boxes_per_class = xywh_boxes[mask]
                    labels_per_class = labels[mask]
                    scores_per_class = scores[mask]
                    indices = cv2.dnn.NMSBoxes(xywh_boxes_per_class, scores_per_class, score_threshold=self.conf, nms_threshold=self.nms)
                    if len(indices):
                        nms_bboxes = np.concatenate((nms_bboxes, boxes_per_class[indices]), axis=0)
                        nms_labels = np.concatenate((nms_labels, labels_per_class[indices]), axis=0)     
            return nms_bboxes, nms_labels                
        bboxes, labels = get_nms_boxes_labels(bboxes, labels, multi_class_nms)
        
        
        def bbox2result(bboxes, labels, num_classes):
            if bboxes.shape[0] == 0:
                return [np.zeros((0, 5), dtype=np.float32) for i in range(num_classes)]
            else:
                return [bboxes[labels == i, :] for i in range(num_classes)]        
        results_list = [bbox2result(bboxes, labels, num_classes=len(self.class_names))]  
        
        if use_all_detector_classes:
            results_list = np.concatenate(results_list[0])
            bboxes = results_list[:, :4]
            scores = results_list[:, 4]
            cls = np.sort(labels).tolist()
        else:
            bboxes = results_list[0][0][:, :4]
            scores = results_list[0][0][:, 4]
            cls = np.zeros_like(scores, dtype=int).tolist()
        
        cls_names = [self.class_names[cl] for cl in cls]
        scores = [round(score, 2) for score in scores.tolist()]
        return (bboxes.astype(int).tolist(), cls, scores, cls_names)