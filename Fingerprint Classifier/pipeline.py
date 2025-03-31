from models import YOLOFingerClassification, YOLOFingerPatternDetection
import os
import cv2
import numpy as np
import random
import time
import math

CLASSIFICATION_THRESHOLD = 0.5
DETECTION_THRESHOLD = 0.25

class DataPipeline:
    def __init__(self, classification_model: str, concentric_whorl_model: str, imploding_whorl_model: str, standard_arch_model: str, loop_model:str):
        self.classification_model = YOLOFingerClassification(classification_model)
        self.concentric_whorl_model = YOLOFingerPatternDetection(concentric_whorl_model)
        self.imploding_whorl_model = YOLOFingerPatternDetection(imploding_whorl_model)
        self.standard_arch_model = YOLOFingerPatternDetection(standard_arch_model)
        self.loop_model = YOLOFingerPatternDetection(loop_model)
    
    def generate_blurred_images(self, image_input_dir:str, image_output_dir: str, num_count: int = 1, ):
        valid_image_formats = (".tif", ".png", ".jpg")
        images = []
        failed_images = []
        if os.path.isdir(image_input_dir):
            for image_name in os.listdir(image_input_dir):
                if image_name.lower().endswith(valid_image_formats):
                    images.append(image_name)
        elif os.path.isfile(image_input_dir):
            images.append(os.path.basename(image_input_dir))
            image_input_dir = os.path.abspath(os.path.join(image_input_dir, ".."))
        
        if not os.path.exists(image_output_dir):
            os.makedirs(image_output_dir)

        for image_name in images:
            img_dir = f"{image_input_dir}/{image_name}"
            
            result = self.classification_model.predict(img_dir, save=False)

            # storing labels for the fingerprint patterns i.e. concentric whorl etc
            labels = result[0].names

            # obtaining the probabilities for each fingerprint pattern in a list
            probabilities = result[0].probs.data.cpu().numpy().tolist()

            # dictionary to store the probabilities as values with labels as keys
            data = {}
            for idx in range(len(labels)):
                data[labels[idx]] = probabilities[idx]
            
            # obtaining the label with highest probability
            label, prob = max(data.items(), key=lambda x: x[1])
            
            if prob < CLASSIFICATION_THRESHOLD:
                failed_images.append(image_name)
                continue
            else:
                match label:
                    case "standard_arch":
                        bounding_box = self.predict_bounding_box(model = self.standard_arch_model, img_dir=img_dir)
                    case "concentric_whorl":
                        bounding_box = self.predict_bounding_box(model = self.concentric_whorl_model, img_dir=img_dir)
                    case "imploding_whorl":
                        bounding_box = self.predict_bounding_box(model = self.imploding_whorl_model, img_dir=img_dir)
                    case "loop":
                        bounding_box = self.predict_bounding_box(model = self.loop_model, img_dir=img_dir, label = label)
                    case _:
                        print("Unknown finger pattern type. Labeller models will need to be updated.")
                        continue
                x1, y1, x2, y2 = bounding_box
                bounding_box_area = (x2-x1)*(y2-y1)
                blur_mask_area = 2/5*bounding_box_area
                radius = math.sqrt(blur_mask_area/math.pi)

                count = 0
                for _ in range(num_count):
                    count += 1
                    lr_axes = random.choice(range(int(radius*0.75), int(radius*1.25)))
                    ud_axes = int(blur_mask_area/math.pi/lr_axes)
                    masked_blur = self.blur_image(img_dir=img_dir, target_box=bounding_box, axes = (lr_axes,ud_axes))
                    filename = f"{image_name}_{count}_blurred_{int(time.time())}"
                    cv2.imwrite(f'{image_output_dir}/{filename}.jpg', masked_blur)

        return failed_images
        
    def predict_bounding_box(self, model: YOLOFingerPatternDetection, img_dir: str):
        try:
            result = model.predict(img_dir, save = False)
            pred_conf = result[0].boxes.conf.tolist()
            target_idx = pred_conf.index(max(pred_conf))

            highest_conf = pred_conf[target_idx]

            if highest_conf > DETECTION_THRESHOLD:
                # [x1, y1, x2, y2]
                bounding_boxes = result[0].boxes.xyxy.tolist()
                target_box = bounding_boxes[target_idx]

                print(f"{highest_conf} - {target_box}")
                
                return target_box
            else:
                print(f"Bounding box confidence is {highest_conf}, which does not meet the minimum threshold level.")
                return None
        except Exception as e:
            print(f"Following error occurred while predicting bounding box: {e}")
            return None

    def blur_image(self, img_dir: str, target_box: list[float], axes: tuple[int]):
        image = cv2.imread(img_dir)

        x1,y1,x2,y2 = target_box
        lr_axes, ud_axes = axes

        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        centerx = int(random.uniform(x1+lr_axes//2, x2-lr_axes//2))
        centery = int(random.uniform(y1+ud_axes//2, y2-ud_axes//2))

        center = (centerx, centery)  # center of image

        cv2.ellipse(mask, center = center, axes = axes, angle=0, startAngle=0, endAngle=360, color=255, thickness=-1)

        blurred = cv2.GaussianBlur(image, (15, 15), 0)

        masked_blur = np.where(mask[:, :, None] == 255, blurred, image)

        return masked_blur


if __name__ == "__main__":
    pipeline = DataPipeline(
        classification_model="Fingerprint Classifier/models/classifier/fingerprint_classifier.pt",
        concentric_whorl_model="Fingerprint Classifier/models/labeller/concentric_whorl_detection.pt",
        imploding_whorl_model="Fingerprint Classifier/models/labeller/imploding_whorl_detection.pt",
        loop_model="Fingerprint Classifier/models/labeller/loop_detection.pt",
        standard_arch_model="Fingerprint Classifier/models/labeller/standard_arch_detection.pt"
    )
    curr_dir = os.getcwd()
    pipeline.generate_blurred_images(image_input_dir=f"{curr_dir}/Fingerprint Classifier/test_images/arch", image_output_dir=f"{curr_dir}/blurred_images", num_count=4)

