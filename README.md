# Wheat Crop Detection using RCNN

This project implements a Region-based Convolutional Neural Network (RCNN) model for detecting wheat crops in images. The model is designed to assist in agricultural monitoring, enabling farmers and researchers to assess crop health and density efficiently.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Training the Model](#training-the-model)
- [Evaluation](#evaluation)
- [Results](#results)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

## Features

- Detects wheat crops in images using an RCNN model.
- Provides bounding boxes around detected crops.
- Supports both training and inference modes.
- Visualizes detection results with bounding boxes.

## Prerequisites

- Python 3.x
- PyTorch
- torchvision
- OpenCV
- NumPy
- Matplotlib
- Other dependencies listed in `requirements.txt`

## Installation

1. **Clone the repository**:
   ```bash
   https://github.com/HashamAkram18/CODXO_Object_Detection_System_Using_Fast_RCNN.git
   ```

3. **Create a virtual environment (optional but recommended)**:
   ```bash
   python -m venv fast-RCNN
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Inference

To run inference on a single image:

```python
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def create_model(num_classes):
    # Load a pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features # Get in_features before replacing
    
    # Replace the pre-trained head with a new one (adjusting for the number of classes)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model
```

### Training the Model

To train the model on your dataset, follow these steps:

1. Prepare your dataset in the required format (images and annotations).
  - [Global Wheat Challenge Dataset](https://www.kaggle.com/datasets/ipythonx/global-wheat-challenge)
3. Use the following command to start training:

```python
python train.py --data_dir path_to_data --num_epochs 50
```

Replace `path_to_data` with the path to your dataset.

## Evaluation

After training, you can evaluate the model's performance 
```
# Specify the number of classes (1 class + background)
num_classes = 2

# Create a model instance
model = create_model(num_classes)

# Load the saved model state dictionary
model.load_state_dict(torch.load('model_weights.pth', map_location=torch.device('cpu')))

```

## Results

The model's performance can be evaluated using metrics such as Mean Average Precision (mAP) and Intersection over Union (IoU). Results will be saved in the `results` directory.

## Limitations

- The model may struggle with images containing overlapping crops or occlusions.
- Performance may vary based on the quality and variety of the training dataset.

## Future Improvements

- Experiment with different architectures (e.g., Faster R-CNN, Mask R-CNN).
- Fine-tune hyperparameters for better accuracy.
- Expand the dataset to include more diverse images of wheat crops.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

---

