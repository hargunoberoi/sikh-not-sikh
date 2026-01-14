# Sikh vs Not Sikh Image Classifier

This project implements a transfer learning approach to classify images of people as "Sikh" or "Not Sikh" using PyTorch and a pre-trained ResNet18 model.

## Setup

1. Install the required packages:

```bash
pip install -r requirements.txt
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

### Command Line

To classify new images using the trained model from the command line, run:

```bash
python inference.py path/to/your/image.jpg [path/to/model.pth]
```

The inference script will load the model and output the prediction with confidence score.

### Web Interface

The project includes a Gradio web interface for easy image classification:

```bash
python app.py
```

This will launch a local web server where you can:

- Upload images through the browser
- See classification results with confidence scores for both classes
- Try sample images if available

The web interface makes it easy for non-technical users to interact with the model.

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
