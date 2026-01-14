import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms
from PIL import Image
import gradio as gr
import os

# Set device for inference
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device for inference")

# Define the same transforms used for validation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_model(model_path='model.pth', num_classes=2):
    """
    Load a trained model for inference.
    """
    # Load the same model architecture used in training
    model = models.resnet18(weights=None)  # No need to download pretrained weights
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    # Load the trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model

def predict_image(img, model, class_names=['Not Sikh', 'Sikh']):
    """
    Predict if a person in an image is Sikh or not.
    
    Args:
        img: The input image (PIL Image or file path)
        model: The trained PyTorch model
        class_names: List of class names
        
    Returns:
        The prediction result with class name and confidence
    """
    try:
        # Handle different input types
        if isinstance(img, str):
            img = Image.open(img).convert('RGB')
        
        # Preprocess the image
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get prediction and confidence for both classes
            probs = probabilities[0].tolist()
            
            # Create result dictionary with all class probabilities
            result = {}
            for i, class_name in enumerate(class_names):
                result[class_name] = float(probs[i])
                
            return result
            
    except Exception as e:
        return {"Error": str(e)}

# Load the model
try:
    model = load_model('model.pth')
    model_loaded = True
except Exception as e:
    print(f"Error loading model: {e}")
    print("Will attempt to load model when making predictions.")
    model = None
    model_loaded = False

def classify_image(image):
    """
    Gradio interface function to classify an uploaded image.
    """
    global model, model_loaded
    
    # Try to load model if not loaded yet
    if not model_loaded:
        try:
            model = load_model('model.pth')
            model_loaded = True
        except Exception as e:
            return {"Error": f"Failed to load model: {str(e)}"}
    
    # Perform prediction
    return predict_image(image, model)

# Create Gradio interface
title = "Sikh or Not Sikh Classifier"
description = """
Upload an image of a person, and the model will classify whether the person is Sikh or not.
The model returns confidence scores for both categories.
"""

# Define the interface
demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=2),
    title=title,
    description=description,
    examples=[
        "data/images/test/hargun.jpeg" if os.path.exists("data/images/test/hargun.jpeg") else None,
        "data/images/test/pavlos.jpeg" if os.path.exists("data/images/test/pavlos.jpeg") else None
    ],
    allow_flagging="never"
)

# Launch the app
if __name__ == "__main__":
    demo.launch(share=False) 