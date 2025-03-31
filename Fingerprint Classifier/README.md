# Steps to starting fingerprint classifier.

## Step 1: Installing necessary packages
1. Run the following line to create a virtual environment.
```
python -m venv venv
```
2. Activate the virtual environment
```
source venv/bin/activate
```
3. Install the necessary dependencies from requirements.txt. Ensure that you are running from "AI-Project" directory, otherwise modify the directory accordingly. 
```
pip install -r "Fingerprint Classifier/requirements.txt"
```
4. If `ultralytics` package cannot be installed due to conflicting dependencies, do:
```
pip install -U ultralytics
pip install -r "Fingerprint Classifier/requirements.txt"
```

## Step 2: Running the pipeline to generate blurred images
1. Consider the following code snipper on how to use`pipeline.py`. Ensure that you have the necessary classifier and labeller models in `Fingerprint Classifier/models`.

```python
from pipeline import DataPipeline

pipeline = DataPipeline(
        classification_model="Fingerprint Classifier/models/classifier/fingerprint_classifier.pt",
        concentric_whorl_model="Fingerprint Classifier/models/labeller/concentric_whorl_detection.pt",
        imploding_whorl_model="Fingerprint Classifier/models/labeller/imploding_whorl_detection.pt",
        loop_model="Fingerprint Classifier/models/labeller/loop_detection.pt",
        standard_arch_model="Fingerprint Classifier/models/labeller/standard_arch_detection.pt"
    )
    # image_input_dir should be a directory to your input images to be blurred
    # image_output_dir should be a directory for the blurred images to be output to
    # num_count refers to the number of blurred images you with to generate per input image
    pipeline.generate_blurred_images(image_input_dir="Fingerprint Classifier/test_images/imploding whorl/imploding_whorl_1.png", image_output_dir="blurred_images", num_count=4)
```

You can also run this from `pipeline.py` directly, refer to the sample code at the bottom of that file. 


