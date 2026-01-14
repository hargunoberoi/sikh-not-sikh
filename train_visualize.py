#%%
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from PIL import Image
import matplotlib.pyplot as plt

#%%

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

data_path = 'images/train'
dataset = datasets.ImageFolder(data_path, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
#%%
sikh_idx = 0
sikh_indices = [i for i, (_, label) in enumerate(dataset.imgs) if label == sikh_idx]
image, label = dataset[sikh_indices[0]]
image = image.unsqueeze(0)

img_display = image[0].permute(1, 2, 0).numpy() * 0.5 + 0.5
plt.imshow(img_display)
plt.title(f"Class: {dataset.classes[label]}")
plt.axis('off')
plt.show()
# %% Build the neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(224 * 224 * 3, 512)
        self.fc2 = nn.Linear(512, 128)
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
# %% Training loop
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 3
dataloader_iter = iter(dataloader)

for epoch in range(num_epochs):
    try:
        images, labels = next(dataloader_iter)
    except StopIteration:
        dataloader_iter = iter(dataloader)
        images, labels = next(dataloader_iter)

    images = images.to(device)
    labels = labels.to(device)

    model.train()
    optimizer.zero_grad()

    logits, probabilities = model(images)
    loss = criterion(logits, labels)

    loss.backward()
    optimizer.step()

    _, predicted = torch.max(probabilities, 1)

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Loss: {loss.item():.4f}")
    print(f"Actual: {dataset.classes[labels.item()]}")
    print(f"Predicted: {dataset.classes[predicted.item()]} ({probabilities[0][predicted.item()].item()*100:.2f}%)")
    print(f"Probabilities: {dataset.classes[0]}={probabilities[0][0].item()*100:.2f}%, {dataset.classes[1]}={probabilities[0][1].item()*100:.2f}%")
    print(f"Correct: {predicted.item() == labels.item()}")

    if epoch < num_epochs - 1:
        input("\nPress Enter for next epoch...")


# %% Full training with batches
batch_dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

model = SimpleNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):
    epoch_loss = 0.0
    correct = 0
    total = 0

    model.train()
    for images, labels in batch_dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits, probabilities = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * images.size(0)
        _, predicted = torch.max(probabilities, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = epoch_loss / len(dataset)
    accuracy = 100 * correct / total
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
#%%
torch.save(model.state_dict(), 'trained_model.pth')

# %%
