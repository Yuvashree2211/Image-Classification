# Convolutional Deep Neural Network for Image Classification


## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and Dataset
Image classification is a fundamental task in computer vision where an input image is assigned to one of several predefined classes. The objective of this experiment is to build and train a Convolutional Neural Network (CNN) using a labeled image dataset and evaluate its performance using accuracy, confusion matrix, and classification report.

# Dataset
For this experiment, the CIFAR-10 dataset is used.

Total Images: 60,000

Training Images: 50,000

Test Images: 10,000

Number of Classes: 10

Image Size: 32 × 32 × 3 (RGB)

<img width="962" height="468" alt="image" src="https://github.com/user-attachments/assets/c5823912-a064-4c12-9811-79bacf35e576" />


## DESIGN STEPS

# STEP 1: Data Preparation
Import required libraries (torch, torchvision, numpy, sklearn).

Load CIFAR-10 dataset.

Normalize the images.

Create DataLoader for training and testing.

# STEP 2: Model Construction
Define CNN class inheriting from nn.Module.

Add convolution, pooling, and fully connected layers.

Define forward propagation.

# STEP 3: Model Training & Evaluation
Define Loss Function (CrossEntropyLoss).

Define Optimizer (Adam).

Train the model for required epochs.

Evaluate using Confusion Matrix and Classification Report.

Test prediction for a new image.


## PROGRAM

### Name: YUVASHREE R
### Register Number: 212224040378
```python
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1=nn.Linear(128*3*3,128)
        self.fc2=nn.Linear(128,64)
        self.fc3=nn.Linear(64,10)




    def forward(self, x):
        x=self.pool(torch.relu(self.conv1(x)))
        x=self.pool(torch.relu(self.conv2(x)))
        x=self.pool(torch.relu(self.conv3(x)))
        x=x.view(x.size(0),-1)
        x=torch.relu(self.fc1(x))
        x=torch.relu(self.fc2(x))
        x=self.fc3(x)
        return x




```

```python
# Initialize model, loss function, and optimizer
model =CNNClassifier()
criterion =nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(), lr=0.001)



```

```python
## Step 3: Train the Model
def train_model(model, train_loader, num_epochs=3):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
          optimizer.zero_grad()
          outputs = model(images)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
          running_loss += loss.item()


        print('Name:YUVASHREE R')
        print('Register Number:  212224040378    ')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')


```

## OUTPUT
### Training Loss per Epoch

<img width="261" height="175" alt="image" src="https://github.com/user-attachments/assets/086dfba3-3098-4615-80cc-1a0aa3c30f11" />




### Confusion Matrix

<img width="724" height="700" alt="image" src="https://github.com/user-attachments/assets/513f3649-2d43-4605-b4d5-1c54751c9199" />



### Classification Report
<img width="440" height="338" alt="image" src="https://github.com/user-attachments/assets/74eb9cc0-ce0d-45ec-8e62-99ca92c8f144" />




### New Sample Data Prediction

<img width="483" height="551" alt="image" src="https://github.com/user-attachments/assets/e1416628-fa87-4f55-973e-c8abc7005a6f" />



## RESULT
The Convolutional Neural Network model was successfully developed and trained using the CIFAR-10 dataset.
