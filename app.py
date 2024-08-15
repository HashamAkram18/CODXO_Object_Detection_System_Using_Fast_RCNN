from flask import Flask, request, jsonify, render_template
import torch
from PIL import Image, ImageDraw
import torchvision
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import io
import numpy as np
import base64

# Function to create the model and replace the classifier head
def create_model(num_classes):
    # Load a pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features # Get in_features before replacing
    
    # Replace the pre-trained head with a new one (adjusting for the number of classes)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

# Specify the number of classes (1 class + background)
num_classes = 2

# Create a model instance
model = create_model(num_classes)

# Load the saved model state dictionary
model.load_state_dict(torch.load('model_weights.pth', map_location=torch.device('cpu')))

# Move the model to the CPU
device = torch.device('cpu')
model.to(device)

# Set the model to evaluation mode
model.eval()

# Define the transformation to apply to the input image
transform = T.Compose([
    T.ToTensor()
])  #  ToTensor() converts the image to a PyTorch tensor and scales pixel values to [0, 1].

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded image file
        file = request.files['image']
        
        # Read the image file into a PIL Image object
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        
        # Transform the image and move it to the device
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Perform inference
        with torch.no_grad():
            output = model(image_tensor)
        
        # Extract the predicted bounding boxes
        boxes = output[0]['boxes'].cpu().numpy()
        
        # Draw bounding boxes on the image
        draw = ImageDraw.Draw(image)
        for box in boxes:
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline="white", width=3)
        
        # Convert image to base64 to embed in HTML
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return render_template('index.html', img_data=img_str)
    
    # Render the HTML template for the upload form
    return render_template('index.html', img_data=None)

if __name__ == '__main__':
    app.run(debug=True, port=9090)
