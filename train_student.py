import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from models import Student

MNIST_DIR = '../mnist/'


# Train student
## Create model
student_model = Student()

## Define Loss
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(student_model.parameters(), lr=0.001, momentum=0.9)

## Load dataset
train_data = torchvision.datasets.MNIST(MNIST_DIR, train=True, download=True,
                                        transform=torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor(), # image to Tensor
                                            torchvision.transforms.Normalize((0.1307,), (0.3081,)) # image, label
                                            ]))
train_loader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True)

for epoch_count in range(5):
    print('epoch: {}'.format(epoch_count))

    ## Optimize parameters
    total_loss = 0.0
    for step_count, (x, y_gt) in enumerate(train_loader):
        # Initialize gradients with 0
        optimizer.zero_grad()

        # Predict
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        y_pred = student_model(x)

        # Compute loss (foward propagation)
        loss = criterion(y_pred, y_gt)
        
        # Compute gradients (backward propagation)
        loss.backward()

        # Update parameters (SGD)
        optimizer.step()

        total_loss += loss.item()
        if step_count % 1000 == 0:
            print('progress: {}\t/ {}\tloss: {}'.format(step_count, len(train_loader), loss.item()))
    
    print('loss: {}'.format(total_loss / len(train_loader)))

## Save model
torch.save(student_model.state_dict(), './data/student.pth')
