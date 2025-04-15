import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms
from PIL import Image
import sys
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

def load_model(model_path, num_classes=2):
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

def predict_image(model, image_path, class_names=['Not Sikh', 'Sikh']):
    """
    Predict if a person in an image is Sikh or not.
    
    Args:
        model: The trained PyTorch model
        image_path: Path to the input image
        class_names: List of class names
        
    Returns:
        The predicted class name and confidence score
    """
    try:
        # Load and preprocess the image
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            
        return class_names[predicted_class.item()], confidence.item()
    
    except Exception as e:
        return f"Error processing image: {str(e)}", 0.0

def main():
    """
    Main function to run inference on image(s).
    """
    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path> [model_path]")
        print("  image_path: Path to the image file to classify")
        print("  model_path: Optional path to the model file (default: model.pth)")
        return
    
    # Get command line arguments
    image_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else 'model.pth'
    
    # Check if files exist
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found")
        return
    
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found")
        return
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    
    # Make prediction
    print(f"Classifying image {image_path}...")
    prediction, confidence = predict_image(model, image_path)
    
    # Display result
    print(f"\nPrediction: {prediction}")
    print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")

if __name__ == "__main__":
    main() 