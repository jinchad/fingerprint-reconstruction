from models import YOLOFingerClassification, YOLOFingerPatternDetection
import os
import cv2
import numpy as np
import random
import time
import math
from typing import Union, List
from blob_masks import blob_mask
from PIL import Image
from numpy.typing import NDArray

CLASSIFICATION_THRESHOLD = 0.5
DETECTION_THRESHOLD = 0.25 
NOISE_THRESHOLD = 127.5 #set between 120 and 135. Higher leads to less noise and vice versa

class DataPipeline:
    def __init__(self, classification_model: str, concentric_whorl_model: str, imploding_whorl_model: str, standard_arch_model: str, loop_model:str):
        self.classification_model = YOLOFingerClassification(classification_model)
        self.concentric_whorl_model = YOLOFingerPatternDetection(concentric_whorl_model)
        self.imploding_whorl_model = YOLOFingerPatternDetection(imploding_whorl_model)
        self.standard_arch_model = YOLOFingerPatternDetection(standard_arch_model)
        self.loop_model = YOLOFingerPatternDetection(loop_model)
    
    def generate_blurred_images(self, image_input_dir:str, image_output_dir: str, num_count: int = 1, noise: bool = True):
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
                        bounding_box = self.predict_bounding_box(model = self.loop_model, img_dir=img_dir)
                    case _:
                        print("Unknown finger pattern type. Labeller models will need to be updated.")
                        continue
                
                count = 0
                x1, y1, x2, y2 = bounding_box
                bounding_box_width = int(abs(x2-x1))
                bounding_box_height = int(abs(y2-y1))

                if noise:
                    max_box_size = max(bounding_box_height, bounding_box_width)
                    for idx in range(num_count):
                        count += 1
                        masked_blur = self.blur_image_with_noise(img_dir=img_dir, box_size = max_box_size)
                        filename = f"{image_name}_{count}_blurred_{int(time.time())}"
                        cv2.imwrite(f'{image_output_dir}/{filename}.jpg', masked_blur)
                
                else:
                    bounding_box_area = (x2-x1)*(y2-y1)
                    blur_mask_area = 0.5*bounding_box_area
                    radius = math.sqrt(blur_mask_area/math.pi)
                    coordinates = DataPipeline.find_best_pts(num_count, x1, x2, y1, y2)

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
        
    def predict_bounding_box(self, model: YOLOFingerPatternDetection, img_dir: str) -> List[float]:
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

        result = image.copy()

        result[mask == 255] = [0, 255, 0]

        return result
    
    def blur_image_with_noise(self, img_dir: str, box_size: int):
        image = cv2.imread(img_dir)
        height, width = image.shape[:2]

        def generate_ridge_like_texture(mask: Image, seed=None):
            mask_array = np.array(mask.convert('L'))
            width, height = mask.size
            rng = np.random.default_rng(seed)
            noise = rng.normal(loc=127, scale=50, size=(height, width)).astype(np.uint8)
            noise_blur = cv2.GaussianBlur(noise, (7, 7), sigmaX=1.5)
            new_blur = find_edge(mask_array, noise_blur)
            return new_blur

        def find_edge(mask: NDArray, noise_blur_array = None):
            mask_width, mask_height = mask.shape
            for row_idx in range(mask_height):
                row = mask[row_idx]
                idx = 0
                while idx < mask_width:
                    num = row[idx]
                    if num != 0:
                        start_idx = idx
                        end_num = row[idx]
                        while end_num != 0:
                            end_num = row[idx]
                            idx += 1
                        end_idx = idx
                        noise_mask_row = noise_blur_array[row_idx]
                        noise_mask_row[:start_idx+1] = 0
                        noise_mask_row[end_idx:] = 0
                        noise_mask_row[start_idx:end_idx] = np.where(noise_mask_row[start_idx:end_idx] >= NOISE_THRESHOLD, 255, 0).astype(np.uint8)
                        noise_blur_array[row_idx] = noise_mask_row
                        break
                    idx += 1
                if noise_blur_array[row_idx][0] != 0 and noise_blur_array[row_idx][-1] != 0:
                    noise_blur_array[row_idx] = 0
            
            return noise_blur_array

        blob = blob_mask(size=box_size)  # PIL image
        ridge_texture = generate_ridge_like_texture(blob)

        new_mask = np.zeros((height, width))
        mask_h, mask_w = ridge_texture.shape    

        start_y = (height - mask_h) // 2
        start_x = (width - mask_w) // 2
        new_mask[start_y:start_y + mask_h, start_x:start_x + mask_w] = ridge_texture

        green_tinted = np.zeros((height, width, 3), dtype=np.float32)
        green_tinted[..., 1] = new_mask.astype(np.float32)  # only green channel gets the texture

        image_float = image.astype(np.float32)
        alpha_mask = new_mask.astype(np.float32) / 255.0
        blended = (green_tinted * alpha_mask[..., None] + image_float * (1 - alpha_mask[..., None])).astype(np.uint8)

        return blended
    
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
    pipeline.generate_blurred_images(image_input_dir=f"{curr_dir}/Fingerprint Classifier/test_images/loop", image_output_dir=f"{curr_dir}/blurred_images", num_count=5)
    


