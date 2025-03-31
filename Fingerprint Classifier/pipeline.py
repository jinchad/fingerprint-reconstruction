from models import YOLOFingerClassification, YOLOFingerPatternDetection
import os
import cv2
import numpy as np
import random
import time

CLASSIFICATION_THRESHOLD = 0.5
DETECTION_THRESHOLD = 0.25

class DataPipeline:
    def __init__(self, classification_model: str, concentric_whorl_model: str, imploding_whorl_model: str, standard_arch_model: str, loop_model:str):
        self.classification_model = YOLOFingerClassification(classification_model)
        self.concentric_whorl_model = YOLOFingerPatternDetection(concentric_whorl_model)
        self.imploding_whorl_model = YOLOFingerPatternDetection(imploding_whorl_model)
        self.standard_arch_model = YOLOFingerPatternDetection(standard_arch_model)
        self.loop_model = YOLOFingerPatternDetection(loop_model)

        self.img_count = 0

    
    def generate_images(self, image_dir:str, num_count: int = 1):
        valid_image_formats = (".tif", ".png", ".jpg")
        images = []
        failed_images = []

        for image_name in os.listdir(image_dir):
            if image_name.lower().endswith(valid_image_formats):
                images.append(image_name)

        for image_name in images:
            img_dir = f"{image_dir}/{image_name}"
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
                        self.predict_bounding_box(model = self.standard_arch_model, img_dir=img_dir, label = label, num_count = num_count)
                        
                    case "concentric_whorl":
                        pass
                    case "imploding_whorl":
                        pass
                    case "loop":
                        pass
                    case _:
                        print("Unknown finger pattern type. Labeller models will need to be updated.")

        return failed_images
        
    def predict_bounding_box(self, model: YOLOFingerPatternDetection, img_dir: str, label: str, num_count: int = 1, mask_size: float = 3/5):
        try:
            result = model.predict(img_dir, save = False)
            pred_conf = result[0].boxes.conf.tolist()
            target_idx = pred_conf.index(max(pred_conf))

            highest_conf = pred_conf[target_idx]

            if highest_conf > DETECTION_THRESHOLD:
                # [x1, y1, x2, y2]
                bounding_boxes = result[0].boxes.xyxy.tolist()
                target_box = bounding_boxes[target_idx]
                x1, y1, x2, y2 = target_box

                print(f"{highest_conf} - {target_box}")
                for _ in range(num_count):
                    lr_radius = random.choice(range(int(1/3*(x2-x1)), int(3/5*(x2-x1)),int(1/10*(x2-x1))))
                    ud_radius = random.choice(range(int(1/3*(y2-y1)), int(3/5*(y2-y1)),int(1/10*(y2-y1))))
                    self.blur_image(img_dir=img_dir, target_box=target_box, axes = (lr_radius,ud_radius), label = label)
            else:
                print(f"Bounding box confidence is {highest_conf}, which does not meet the minimum threshold level.")
        except Exception as e:
            print(f"Following error occurred while predicting bounding box: {e}")

    def blur_image(self, img_dir: str, target_box: list[float], axes: tuple[int], label: str):
        self.img_count += 1
        image = cv2.imread(img_dir)

        x1,y1,x2,y2 = target_box
        lr_radius, ud_radius = axes

        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        centerx = int(random.uniform(x1+lr_radius//2, x2-lr_radius//2))
        centery = int(random.uniform(y1+ud_radius//2, y2-ud_radius//2))

        center = (centerx, centery)  # center of image

        cv2.ellipse(mask, center = center, axes = axes, angle=0, startAngle=0, endAngle=360, color=255, thickness=-1)

        blurred = cv2.GaussianBlur(image, (15, 15), 0)

        masked_blur = np.where(mask[:, :, None] == 255, blurred, image)

        filename = f"blurred_{label}_{self.img_count}_{int(time.time())}"

        cv2.imwrite(f'{filename}.jpg', masked_blur)


if __name__ == "__main__":
    pipeline = DataPipeline(
        classification_model="Fingerprint Classifier/models/classifier/fingerprint_classifier.pt",
        concentric_whorl_model="Fingerprint Classifier/models/labeller/concentric_whorl_detection.pt",
        imploding_whorl_model="Fingerprint Classifier/models/labeller/imploding_whorl_detection.pt",
        loop_model="Fingerprint Classifier/models/labeller/loop_detection.pt",
        standard_arch_model="Fingerprint Classifier/models/labeller/standard_arch_detection.pt"
    )

    pipeline.generate_images("Fingerprint Classifier/test_images/arch", num_count=4)

