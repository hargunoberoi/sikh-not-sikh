# Sikh vs Not Sikh Image Classifier

This project implements a transfer learning approach to classify images of people as "Sikh" or "Not Sikh" using PyTorch and a pre-trained ResNet18 model.

## Setup

1. Install the required packages:

```bash
pip install torch torchvision pillow matplotlib numpy
```

2. Make sure your dataset is organized in the following structure:

```
data/
    images/
        train/
            Sikh/
                image1.jpg
                image2.jpg
                ...
            Not Sikh/
                image1.jpg
                image2.jpg
                ...
        test/
            test_image1.jpg
            test_image2.jpg
            ...
```

## Training

To train the model, run:

```bash
python sikh_classification.py
```

The script will:

1. Create a training and validation split from the training data
2. Fine-tune a pre-trained ResNet18 model
3. Save the best model as `model.pth`
4. Visualize model predictions on validation data
5. Test the model on test images

Training parameters:

- Batch size: 4
- Learning rate: 0.001 with SGD optimizer and momentum 0.9
- Learning rate scheduler: StepLR with step size 7 and gamma 0.1
- Number of epochs: 25

## Inference

To classify new images using the trained model, run:

```bash
python inference.py path/to/your/image.jpg [path/to/model.pth]
```

The inference script will load the model and output the prediction with confidence score.

## Model Architecture

The model uses a ResNet18 architecture pre-trained on ImageNet. The final fully connected layer is modified to classify between 2 classes (Sikh and Not Sikh).

## Data Preprocessing

Images are preprocessed using the following transformations:

**Training data:**

- Random resize and crop to 224x224
- Random horizontal flip
- Normalization with ImageNet mean and std

**Validation/Test data:**

- Resize to 256x256
- Center crop to 224x224
- Normalization with ImageNet mean and std
