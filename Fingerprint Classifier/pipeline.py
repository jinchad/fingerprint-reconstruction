from ultralytics import YOLO

model = YOLO("/Users/jin/Documents/GitHub/AI-Project/Fingerprint Classifier/models/labeller/imploding_whorl_detection.pt")

model.predict("/Users/jin/Documents/GitHub/AI-Project/Fingerprint Classifier/test_images/whorl", save = True)

