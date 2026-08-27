# 🐶🐱 Cat vs Dog Image Classifier

A deep learning project that classifies images into two categories: **Cat** or **Dog** using **Transfer Learning with ResNet18** and **PyTorch**.

The project covers the complete machine learning workflow, from dataset preparation and model training to evaluation and prediction on new images.

---

## 📌 Project Overview

The goal of this project is to build an image classification model that can automatically determine whether an input image contains a **cat** or a **dog**.

### Workflow

```text
Image
  ↓
Data Preprocessing
  ↓
Data Augmentation
  ↓
Pretrained ResNet18
  ↓
Transfer Learning
  ↓
Model Training
  ↓
Model Evaluation
  ↓
Cat / Dog Prediction
```

---

## 🎯 Objectives

* Learn the fundamentals of image classification.
* Understand how Convolutional Neural Networks (CNNs) are used for computer vision.
* Apply transfer learning using a pretrained ResNet18 model.
* Train a binary image classification model.
* Evaluate model performance using accuracy and other metrics.
* Build a model that can classify previously unseen images.
* Prepare the trained model for deployment in a simple web application.

---

## 🛠️ Technologies

| Technology   | Purpose                                         |
| ------------ | ----------------------------------------------- |
| Python       | Programming language                            |
| PyTorch      | Deep learning framework                         |
| Torchvision  | Computer vision utilities and pretrained models |
| Google Colab | Development and GPU training environment        |
| Matplotlib   | Visualization                                   |
| Scikit-learn | Model evaluation                                |
| Streamlit    | Web application                                 |

---

## 📂 Project Structure

```text
cat-dog-classifier/
│
├── dataset/
│   ├── train/
│   │   ├── cat/
│   │   └── dog/
│   │
│   ├── validation/
│   │   ├── cat/
│   │   └── dog/
│   │
│   └── test/
│       ├── cat/
│       └── dog/
│
├── notebooks/
│   └── training.ipynb
│
├── models/
│   └── cat_dog_model.pth
│
├── results/
│   ├── confusion_matrix.png
│   ├── training_loss.png
│   └── accuracy.png
│
├── app.py
├── requirements.txt
└── README.md
```

---

# 1. Dataset

The dataset contains images belonging to two classes:

```text
Cat
Dog
```

The dataset is divided into three subsets:

* **Training set** — used to train the model.
* **Validation set** — used to monitor model performance during development.
* **Test set** — used for the final evaluation.

### Recommended Split

```text
Training      70%
Validation    15%
Testing       15%
```

For example, with 10,000 images:

```text
Train        7,000 images
Validation   1,500 images
Test         1,500 images
```

The test images should not be used during training.

---

# 2. Environment Setup

This project can be developed using **Google Colab**, especially when GPU acceleration is needed.

Check whether a GPU is available:

```python
import torch

print(torch.cuda.is_available())
```

If the output is:

```text
True
```

the model can use CUDA acceleration.

Set the computing device:

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(device)
```

---

# 3. Install Dependencies

Install the required Python packages:

```bash
pip install torch torchvision matplotlib scikit-learn
```

For the web application:

```bash
pip install streamlit
```

---

# 4. Data Preprocessing

Images may have different sizes, so they need to be converted into a consistent format before being passed to the neural network.

This project resizes images to:

```text
224 × 224 pixels
```

Basic preprocessing:

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
```

---

# 5. Data Augmentation

Data augmentation is applied to the training images to improve generalization and reduce overfitting.

Example:

```python
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])
```

The model may therefore see slightly different versions of the same image during training.

For example:

```text
Original Image
      ↓
Horizontal Flip
      ↓
Small Rotation
      ↓
Training Image
```

---

# 6. Load the Dataset

PyTorch's `ImageFolder` can automatically assign labels based on the folder names.

```python
from torchvision import datasets

train_dataset = datasets.ImageFolder(
    "dataset/train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    "dataset/validation",
    transform=transform
)

test_dataset = datasets.ImageFolder(
    "dataset/test",
    transform=transform
)
```

Check the class names:

```python
print(train_dataset.classes)
```

Expected output:

```text
['cat', 'dog']
```

The labels are generally:

```text
cat → 0
dog → 1
```

---

# 7. Create DataLoaders

The dataset is loaded in batches using PyTorch's `DataLoader`.

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)
```

### Parameters

* `batch_size=32` — processes 32 images at a time.
* `shuffle=True` — randomly shuffles the training data.
* `shuffle=False` — keeps validation and test data ordered.

---

# 8. Model — ResNet18

This project uses **ResNet18** with transfer learning.

Instead of training a neural network completely from scratch, a pretrained model is used as the starting point.

```python
from torchvision.models import resnet18, ResNet18_Weights

weights = ResNet18_Weights.DEFAULT

model = resnet18(weights=weights)
```

The original ResNet18 classifier is replaced with a new classifier for two classes:

```python
import torch.nn as nn

model.fc = nn.Linear(
    model.fc.in_features,
    2
)
```

The model is then moved to the selected device:

```python
model = model.to(device)
```

---

# 9. Loss Function

The model uses Cross Entropy Loss for binary classification with two output classes.

```python
criterion = nn.CrossEntropyLoss()
```

The loss measures how different the model's prediction is from the correct label.

---

# 10. Optimizer

Adam is used to optimize the neural network parameters.

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)
```

The learning rate controls how much the model's parameters are updated during training.

---

# 11. Train the Model

The model is trained for multiple epochs.

```python
num_epochs = 10

for epoch in range(num_epochs):

    model.train()

    running_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print(
        f"Epoch {epoch+1}/{num_epochs}, "
        f"Loss: {running_loss/len(train_loader):.4f}"
    )
```

During training, the model repeatedly:

```text
Input Images
     ↓
Prediction
     ↓
Calculate Loss
     ↓
Backpropagation
     ↓
Update Weights
     ↓
Next Batch
```

The objective is to gradually reduce the training loss.

---

# 12. Validation

Validation data is used to measure how well the model performs on images that were not used to update the model's weights.

```python
model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

accuracy = correct / total

print("Validation Accuracy:", accuracy)
```

---

# 13. Model Evaluation

After training is complete, the test dataset is used for final evaluation.

```python
model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

accuracy = correct / total

print("Test Accuracy:", accuracy)
```

Example:

```text
Test Accuracy: 0.95
```

This means that the model correctly classified approximately **95% of the test images**.

> The actual performance depends on the dataset, training configuration, and quality of the images.

---

# 14. Evaluation Metrics

Accuracy alone does not provide the complete picture of model performance.

The following metrics can also be calculated:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

### Confusion Matrix

A confusion matrix shows which classes the model gets right or wrong.

Example:

```text
                 Predicted
                Cat     Dog

Actual Cat      480      20
Actual Dog       30     470
```

This allows us to identify whether the model has difficulty distinguishing certain images.

---

# 15. Prediction on a New Image

Once the model has been trained, it can classify a new image.

```python
from PIL import Image

image = Image.open("my_cat.jpg")

image_tensor = transform(image)

image_tensor = image_tensor.unsqueeze(0)

image_tensor = image_tensor.to(device)
```

Run the image through the model:

```python
model.eval()

with torch.no_grad():

    output = model(image_tensor)

    probabilities = torch.softmax(output, dim=1)

    prediction = torch.argmax(
        probabilities,
        dim=1
    )
```

The result is:

```text
0 → Cat
1 → Dog
```

---

# 16. Prediction Confidence

The model's probability can also be displayed.

```python
cat_probability = probabilities[0][0].item()
dog_probability = probabilities[0][1].item()

print("Cat:", cat_probability)
print("Dog:", dog_probability)
```

Example:

```text
Cat: 0.973
Dog: 0.027
```

The application can display:

```text
Prediction: CAT 🐱
Confidence: 97.3%
```

---

# 17. Save the Trained Model

The trained model can be saved so that it does not need to be trained again.

```python
torch.save(
    model.state_dict(),
    "cat_dog_model.pth"
)
```

The resulting file:

```text
models/cat_dog_model.pth
```

contains the learned model parameters.

---

# 18. Web Application

The trained model can be integrated into a simple web application using **Streamlit**.

The application should allow users to:

1. Upload an image.
2. Process the image.
3. Run the image through the trained model.
4. Display the predicted class.
5. Display the prediction confidence.

Example interface:

```text
┌──────────────────────────────┐
│     🐶🐱 CAT vs DOG AI       │
├──────────────────────────────┤
│                              │
│       Upload an image        │
│                              │
│      [ Choose File ]         │
│                              │
│            🐱                │
│                              │
│      Prediction: CAT         │
│      Confidence: 97.3%       │
│                              │
└──────────────────────────────┘
```

Run the application:

```bash
streamlit run app.py
```

---

# 📊 Results

The following results should be recorded after training:

| Metric              | Result |
| ------------------- | -----: |
| Training Accuracy   |    TBD |
| Validation Accuracy |    TBD |
| Test Accuracy       |    TBD |
| Precision           |    TBD |
| Recall              |    TBD |
| F1-score            |    TBD |

Replace `TBD` with the actual results from the trained model.

---

# 📈 Visualizations

The project should include:

### Training Loss

Shows whether the model's training loss decreases over time.

```text
Loss
 │\
 │ \
 │  \
 │   \____
 │
 └──────────── Epoch
```

### Accuracy

Shows how model performance changes during training.

### Confusion Matrix

Shows the number of correct and incorrect predictions for each class.

These visualizations can help identify problems such as **overfitting** or poor generalization.

---

# ⚠️ Limitations

This model has several limitations:

* It only recognizes two categories.
* Performance depends heavily on dataset quality.
* Unusual images may produce incorrect predictions.
* Images containing multiple animals may be difficult to classify.
* Poor lighting, occlusion, or unusual viewpoints can reduce accuracy.
* A high test accuracy does not guarantee good performance on every real-world image.

---

# 🚀 Future Improvements

Possible improvements include:

### 1. More Classes

Expand the classifier:

```text
Cat
Dog
Rabbit
Bird
Horse
...
```

### 2. Dog Breed Classification

Instead of:

```text
Dog
```

classify:

```text
Golden Retriever
Poodle
Husky
Shiba Inu
German Shepherd
...
```

### 3. Object Detection

Instead of only classifying the entire image, detect where the animals are located.

```text
┌────────────────────────────┐
│                            │
│   ┌─────────┐              │
│   │   🐶    │  Dog         │
│   └─────────┘              │
│                            │
│          ┌──────┐          │
│          │  🐱  │  Cat     │
│          └──────┘          │
│                            │
└────────────────────────────┘
```

Possible models include YOLO and Faster R-CNN.

### 4. Model Optimization

Experiment with:

* ResNet18
* ResNet50
* MobileNet
* EfficientNet

and compare their accuracy, speed, and model size.

### 5. Deploy the Model

The model can eventually be deployed as:

* Web application
* REST API
* Mobile application
* Cloud service
* Edge AI application

---

# 📚 Learning Outcomes

Through this project, the following concepts are practiced:

```text
Python
   ↓
PyTorch
   ↓
Image Processing
   ↓
CNN
   ↓
Transfer Learning
   ↓
Model Training
   ↓
Validation
   ↓
Evaluation
   ↓
Model Deployment
```

The project provides a practical introduction to **Machine Learning, Deep Learning, and Computer Vision**.

---

# 📝 Conclusion

This project demonstrates an end-to-end image classification pipeline using PyTorch and transfer learning.

A pretrained **ResNet18** model is adapted to classify images into two categories:

```text
🐱 Cat
🐶 Dog
```

The trained model can then be evaluated, saved, and integrated into a web application for real-world image classification.

---

## 👤 Author

**Kanonlas Rattanapak**

into AI right now 

---

## 📄 License

This project is intended for educational and research purposes.
