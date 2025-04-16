from models import YOLOFingerClassification, YOLOFingerPatternDetection
import os
import cv2
import numpy as np
import random
import time
import math
from typing import Union

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
                else:
                    failed_images.append(image_name)

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
                blur_mask_area = 0.5*bounding_box_area
                radius = math.sqrt(blur_mask_area/math.pi)
                coordinates = DataPipeline.find_best_pts(num_count, x1, x2, y1, y2)

                count = 0

                for idx in range(num_count):
                    count += 1
                    coor = tuple(map(lambda x: int(x), coordinates[idx]))

                    # left-right radius of the ellipse
                    lr_axes = random.choice(range(int(radius*0.75), int(radius*1.25)))

                    # up-down radius of the ellipse
                    ud_axes = int(blur_mask_area/math.pi/lr_axes)

                    masked_blur = self.blur_image(img_dir=img_dir, axes = (lr_axes,ud_axes), centre = coor)
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

    def blur_image(self, img_dir: str, axes: tuple[int, int], centre: tuple[int, int]):
        image = cv2.imread(img_dir)

        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        cv2.ellipse(mask, center = centre, axes = axes, angle=0, startAngle=0, endAngle=360, color=255, thickness=-1)

        blurred = cv2.GaussianBlur(image, (15, 15), 0)

        masked_blur = np.where(mask[:, :, None] == 255, blurred, image)

        result = image.copy()

        result[mask == 255] = [0, 255, 0]

        return result
    
    @staticmethod
    def find_best_pts(n: int, x1, x2, y1, y2):
        coor = DataPipeline.obtain_coordinates(n, x1, x2, y1, y2)
        centre_coor = (x1+abs(x2-x1)/2, y1+abs(y2-y1)/2)

        pts = []
        for idx in range(len(coor)):
            coor1 = coor[idx]
            score = 0
            for i in range(len(coor)):
                coor2 = coor[i]
                score += DataPipeline.calculate_distance(coor1=coor1, coor2=coor2)
            pts.append((score, coor1))
        
        pts.sort(key = lambda pt: pt[0], reverse=True)

        best_pts = [centre_coor]
        for _ in range(n-1):
            for idx in range(len(pts)):
                pt_coor = pts[idx][1]
                new_distance = 0 
                for best_pt in best_pts:
                    new_distance += DataPipeline.calculate_distance(best_pt, pt_coor)
                pts[idx] = (new_distance, pt_coor)
            pts.sort(key = lambda pt: pt[0], reverse=True)
            best_new_coor = pts.pop(0)[1]
            best_pts.append(best_new_coor)
            
        return best_pts
    
    @staticmethod
    def obtain_coordinates(n: int, x1, x2, y1, y2):
        bounding_box_length = abs(x2-x1)
        bounding_box_height = abs(y2-y1)
        coor = set()
        pts_per_row = math.ceil(np.sqrt(n))
        if pts_per_row > 2:
            for i in range(1,pts_per_row-1):
                for j in range(1, pts_per_row-1):
                    coor.add((i/(pts_per_row-1), j/(pts_per_row-1)))
        for i in range(pts_per_row):
            coor.add((0, i/(pts_per_row-1)))
            coor.add((1, i/(pts_per_row-1)))
            coor.add((i/(pts_per_row-1), 0))
            coor.add((i/(pts_per_row-1), 1))

        print(f"Number of coordinates generateed: {len(coor)}")

        coor = list(coor)

        for idx in range(len(coor)):
            c = coor[idx]
            x_coor = x1 + c[0] * bounding_box_length
            y_coor = y1 + c[1] * bounding_box_height
            coor[idx] = (x_coor, y_coor)

        return coor

    @staticmethod
    def calculate_distance(coor1: tuple[Union[int, float], Union[int, float]], 
                        coor2: tuple[Union[int, float], Union[int, float]]
                        ) -> float:
        return np.sqrt((coor1[0]-coor2[0])**2+(coor1[1]-coor2[1])**2)


if __name__ == "__main__":
    pipeline = DataPipeline(
        classification_model="Fingerprint Classifier/models/classifier/fingerprint_classifier.pt",
        concentric_whorl_model="Fingerprint Classifier/models/labeller/concentric_whorl_detection.pt",
        imploding_whorl_model="Fingerprint Classifier/models/labeller/imploding_whorl_detection.pt",
        loop_model="Fingerprint Classifier/models/labeller/loop_detection.pt",
        standard_arch_model="Fingerprint Classifier/models/labeller/standard_arch_detection.pt"
    )
    curr_dir = os.getcwd()
    pipeline.generate_blurred_images(image_input_dir=f"{curr_dir}/Fingerprint Classifier/test_images/imploding whorl", image_output_dir=f"{curr_dir}/blurred_images", num_count=5)
    
