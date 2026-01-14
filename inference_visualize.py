#%%
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(224 * 224 * 3,1024) 
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 224 * 224 * 3)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        logits = self.fc3(x)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        return logits, probabilities

model = SimpleNet().to(device)
model.load_state_dict(torch.load('trained_model.pth', map_location=device))
model.eval()

#%%
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

class_names = ['not sikh', 'sikh']

#%%
test_image_path = None
for root, dirs, files in os.walk('images'):
    for file in files:
        if file.endswith(('.jpg', '.jpeg', '.png')):
            test_image_path = os.path.join(root, file)
            break
    if test_image_path:
        break

image = Image.open(test_image_path).convert('RGB')
image_tensor = transform(image).unsqueeze(0).to(device)
#%%
with torch.no_grad():
    logits, probabilities = model(image_tensor)
    _, predicted = torch.max(probabilities, 1)

plt.imshow(image)
plt.title(f"Predicted: {class_names[predicted.item()]} ({probabilities[0][predicted.item()].item()*100:.2f}%)")
plt.axis('off')
plt.show()

print(f"Prediction: {class_names[predicted.item()]}")
print(f"Confidence: {probabilities[0][predicted.item()].item()*100:.2f}%")
print(f"Probabilities: {class_names[0]}={probabilities[0][0].item()*100:.2f}%, {class_names[1]}={probabilities[0][1].item()*100:.2f}%")

# %%