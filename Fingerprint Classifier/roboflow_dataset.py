import os
import cv2
from roboflow import Roboflow
import re
import yaml
import shutil
import copy
from dotenv import load_dotenv

load_dotenv()

class RoboflowDataset:
    ### UPDATE THIS IF DIFFERENT CLASSES
    labels = {   
            "concentric_whorl": "0",
            "imploding_whorl": "1",
            "loop": "2",
            "standard_arch": "3"
        }
    
    def __init__(self, rf_project: str = "fingerprint-pattern-detection-vmh4p", rf_dataset_ver: int = 1) -> None:
        self.rf_key = os.environ.get("RF_KEY", "")
        self.rf_workspace = os.environ.get("RF_WORKSPACE", "")
        self.rf_project = rf_project
        self.rf_dataset_ver = rf_dataset_ver
        self.curr_dir = os.getcwd()
        parent_dir = os.path.abspath(os.path.join(self.curr_dir, ".."))
        self.dataset_dir = os.path.abspath(os.path.join(parent_dir, f"datasets/{self.rf_project}/v{self.rf_dataset_ver}"))
        
    # Saturating Image to further enhance the fingerprint ridges
    @staticmethod
    def saturate_image(image_file_path: str) -> None:
        image = cv2.imread(image_file_path, cv2.IMREAD_GRAYSCALE)
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        os.remove(image_file_path)
        cv2.imwrite(f"{image_file_path}", enhanced)

    def filter_images(self, data_type: str, class_name: str) -> None:
        data_type_dir = f"{self.dataset_dir}/{class_name}_dataset_v{self.rf_dataset_ver}/{data_type}"

        label_dir = os.path.abspath(os.path.join(f"{self.dataset_dir}/original_dataset/{data_type}/", "labels"))
        image_dir = os.path.abspath(os.path.join(f"{self.dataset_dir}/original_dataset/{data_type}/", "images"))
        label_num = RoboflowDataset.labels[class_name]

        for label_file_name in os.listdir(label_dir):
            original_label_file_path = os.path.join(label_dir, label_file_name)
            try:
                with open(original_label_file_path, "r") as label_file:
                    lines = label_file.read()
                image_file_name = re.sub(r"\.[^.]+$", "", label_file_name)+".jpg"
                original_image_file_path = os.path.join(image_dir, image_file_name)

                if lines[0] == label_num:
                    parts = lines.strip().split()
                    parts[0] = "0"
                    l1 = " ".join(parts)                    
                    with open(original_label_file_path, "w") as label_file:
                        label_file.write(l1)
                    shutil.move(original_label_file_path, f"{data_type_dir}/labels/{label_file_name}")
                    new_image_file_path = f"{data_type_dir}/images/{image_file_name}"
                    shutil.move(original_image_file_path, new_image_file_path)
                    RoboflowDataset.saturate_image(image_file_path=new_image_file_path)

            except IndexError: # catches any images labelled as "null" on Roboflow
                    os.remove(original_label_file_path)
                    image_file_name = re.sub(r"\.[^.]+$", "", label_file_name)+".jpg"
                    original_image_file_path = os.path.join(image_dir, image_file_name)
                    os.remove(original_image_file_path)
            
    def download_dataset(self, export_type: str = "yolov8", dataset_dir: str = None) -> None:
        if not dataset_dir:
            dataset_dir = self.dataset_dir
        if os.path.exists(dataset_dir):
            shutil.rmtree(dataset_dir)
        rf = Roboflow(api_key=self.rf_key)
        project = rf.workspace(self.rf_workspace).project(self.rf_project)
        project.version(self.rf_dataset_ver).download(export_type, f"{dataset_dir}/original_dataset")
    
    def delete_original_dataset(self) -> None:
        original_dataset_dir = f"{self.dataset_dir}/original_dataset"
        if os.path.exists(original_dataset_dir):
            shutil.rmtree(original_dataset_dir)

    def create_class_dir(self, class_name:str) -> None:
        class_dir = os.path.abspath(os.path.join(self.dataset_dir, f"{class_name}_dataset_v{self.rf_dataset_ver}"))
        if not os.path.exists(class_dir):
            for i in ["test", "train", "valid"]:
                os.makedirs(f"{class_dir}/{i}/images")
                os.makedirs(f"{class_dir}/{i}/labels")
                self.filter_images(class_name=class_name, data_type=i)
            self.create_class_yaml(class_name= class_name, class_dir=class_dir)
    
    def create_class_yaml(self, class_name: str, class_dir: str) -> None:
        og_yaml_dir = os.path.abspath(os.path.join(self.dataset_dir, "original_dataset/data.yaml"))

        with open(og_yaml_dir, "r") as f:
            yaml_data = yaml.safe_load(f)
        modified_yaml_data = copy.deepcopy(yaml_data)
        modified_yaml_data['names'] = [class_name]
        modified_yaml_data['nc'] = 1

        with open(f"{class_dir}/data.yaml", 'w') as f:
            yaml.dump(modified_yaml_data, f, sort_keys=False)    

if __name__ == "__main__":
    rf_dataset = RoboflowDataset(rf_dataset_ver = 6)
    rf_dataset.download_dataset()
    for c in RoboflowDataset.labels.keys():
        rf_dataset.create_class_dir(c)
    rf_dataset.delete_original_dataset()

    # rf_dataset = RoboflowDataset(rf_project="fingerprint-classification-un8qd", rf_dataset_ver= 2)
    # rf_dataset.download_dataset(export_type="folder")