# Convolutional Deep Neural Network for Image Classification

## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and Dataset

Include the Problem Statement and Dataset.

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:
Import the necessary libraries such as NumPy, Matplotlib, and PyTorch.

### STEP 2:
Load and preprocess the dataset:

Resize images to a fixed size (128×128). Normalize pixel values to a range between 0 and 1. Convert labels into numerical format if necessary.

### STEP 3:
Define the CNN Architecture, which includes:

Input Layer: Shape (8,128,128) Convolutional Layer 1: 8 filters, kernel size (16×16), ReLU activation Max-Pooling Layer 1: Pool size (2×2) Convolutional Layer 2: 24 filters, kernel size (8×8), ReLU activation Max-Pooling Layer 2: Pool size (2×2) Fully Connected (Dense) Layer: First Dense Layer with 256 neurons Second Dense Layer with 128 neurons Output Layer for classification

### STEP 4:
Define the loss function (e.g., Cross-Entropy Loss for classification) and optimizer (e.g., Adam or SGD).

### STEP 5:
Train the model by passing training data through the network, calculating the loss, and updating the weights using backpropagation.

### STEP 6:
Evaluate the trained model on the test dataset using accuracy, confusion matrix, and other performance metrics.

### STEP 7:
Make predictions on new images and analyze the results.


## PROGRAM

### Name: YUVASHREE R
### Register Number: 212224040378
```python
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

     def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x




```

```python
# Initialize the Model, Loss Function, and Optimizer
model = CNNClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


```

```python
# Train the Model
def train_model(model, train_loader, num_epochs=3):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        
        print('Name: YUVASHREE R       ')
        print('Register Number:  212224040378     ')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

```

## OUTPUT
### Training Loss per Epoch

<img width="689" height="772" alt="image" src="https://github.com/user-attachments/assets/e221bb8e-87a2-43e6-81c6-6dc27952f307" />



### Confusion Matrix

<img width="709" height="607" alt="image" src="https://github.com/user-attachments/assets/396e1b1e-c5f1-4878-b47f-9cbba42193e2" />


### Classification Report
<img width="437" height="347" alt="image" src="https://github.com/user-attachments/assets/945da73f-de01-4b71-83d3-6697cbd4f475" />



### New Sample Data Prediction

<img width="421" height="489" alt="image" src="https://github.com/user-attachments/assets/4b5f67eb-6a76-4ccb-a346-5b48804160fe" />


## RESULT
Thus the development of a convolutional deep neural network for image classification is executed successfully.
