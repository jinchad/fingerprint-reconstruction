from ultralytics import YOLO
from abc import ABC, abstractmethod
from roboflow_dataset import RoboflowDataset
import os

class YOLOModel(ABC):
    def __init__(self, model_name: str):
        self.model = YOLO(model_name)
    
    @abstractmethod
    def train(self, dataset_dir: str):
        pass

    @abstractmethod
    def predict(self, image_dir: str):
        pass

    @abstractmethod
    def load_dataset(self):
        pass

class YOLOFingerClassification(YOLOModel):
    def __init__(self, model_name: str, project_name: str = "fingerprint-pattern-classifier"):
        super().__init__(model_name)
        self.project_name = project_name
    
    def train(self, version_num: int, dataset_dir: str, epochs: int = 20, imgsz: int = 256, batch: int = 32, dropout: float = 0.2, save_period: int = 1, patience: int = 5) -> None:
        self.load_dataset(version_num=version_num, dataset_dir=dataset_dir)

        self.model.train(
            data=dataset_dir,  # auto set by roboflow
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            dropout = dropout,
            save_period = save_period,
            patience = patience,
            val = True,
            name="fingerprint-classification"
        )
    
    def predict(self, image_dir: str, save: bool = True) -> None:
        return self.model.predict(image_dir, save = save)
    
    def load_dataset(self, version_num: int, dataset_dir: str) -> None:
        if not os.path.exists(dataset_dir):
            print("Downloading the dataset")
            rf = RoboflowDataset(rf_project=self.project_name, rf_dataset_ver= version_num)
            dataset_download_dir = os.path.abspath(os.path.join(dataset_dir, ".."))
            rf.download_dataset(export_type="folder", dataset_dir=dataset_download_dir)
        

class YOLOFingerPatternDetection(YOLOModel):
    def __init__(self, model_name: str, project_name: str = "fingerprint-pattern-detection-vmh4p"):
        super().__init__(model_name)
        self.project_name = project_name

    def train(self, version_num: int, dataset_dir: str, pattern: str, epochs: int = 50, imgsz: int = 256, batch: int = 32, dropout: float = 0.2, save_period: int = 4, patience: int = 20) -> None:
        self.load_dataset(version_num=version_num, dataset_dir=dataset_dir)

        self.model.train(
            data=dataset_dir,  # auto set by roboflow
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            dropout = dropout,
            save_period = save_period,
            patience = patience,
            val = True,
            name="fingerprint-classification"
        )
    
    def predict(self, image_dir: str, save: bool = True) -> None:
        return self.model.predict(image_dir, save = save)
    
    def load_dataset(self, version_num: int, dataset_dir: str) -> None:
        if not os.path.exists(dataset_dir):
            rf_dataset = RoboflowDataset(rf_project=self.project_name, rf_dataset_ver= version_num)
            # move from data.yaml to version folder
            dataset_download_dir = os.path.abspath(os.path.join(dataset_dir, ".."))

            rf_dataset.download_dataset(export_type="yolov8", dataset_dir=dataset_download_dir)
            for c in RoboflowDataset.labels.keys():
                rf_dataset.create_class_dir(c)
            rf_dataset.delete_original_dataset()



if __name__ == "__main__":
    # y = YOLOFingerPatternDetection("yolov8n.pt")
    # fingerprint_pattern = "concentric_whorl"
    # version_num = 6
    # y.train(dataset_dir=f"/Users/jin/Documents/GitHub/datasets/fingerprint-pattern-detection-vmh4p/v{version_num}/{fingerprint_pattern}_dataset_v{version_num}/data.yaml", version_num=version_num, pattern = fingerprint_pattern)

    y = YOLOFingerPatternDetection("Fingerprint Classifier/models/classifier/fingerprint_classifier.pt")
    y.predict("/Users/jin/Documents/GitHub/AI-Project/Fingerprint Classifier/test_images/imploding whorl/imploding_whorl_1.png")
