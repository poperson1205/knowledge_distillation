import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import scheduler

from models import Teacher

MNIST_DIR = '../mnist/'

# Create gpu device
device = torch.device('cuda')
print(device)

# Create model
teacher_model = Teacher()

# Transfer
teacher_model.to(device)

# Define Loss
criterion = nn.CrossEntropyLoss(reduction='mean')

# Define optimizer
optimizer = optim.SGD(teacher_model.parameters(), lr=0.1)

# Define schedule for learning rate and momentum
lr_init = 0.1
gamma = 0.998
lrs = np.zeros(shape=(3000,))
lr = lr_init
for step in range(3000):
    lrs[step] = lr
    lr *= gamma
momentums = np.concatenate([np.linspace(0.5, 0.99, 500), np.full(shape=(2500,), fill_value=0.99)])
list_lr_momentum_scheduler = scheduler.ListScheduler(optimizer, lrs=lrs, momentums=momentums)

# Load dataset
train_data = torchvision.datasets.MNIST(MNIST_DIR, train=True, download=True,
                                        transform=torchvision.transforms.Compose([
                                            torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                                            torchvision.transforms.ToTensor(), # image to Tensor
                                            torchvision.transforms.Normalize((0.1307,), (0.3081,)) # image, label
                                            ]))

train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)

teacher_model.train()
for epoch_count in range(3000):
    print('epoch: {}'.format(epoch_count))

    ## Optimize parameters
    total_loss = 0.0
    for step_count, (x, y_gt) in enumerate(train_loader):
        # Initialize gradients with 0
        optimizer.zero_grad()

        # Transfer device
        x = x.to(device)
        y_gt = y_gt.to(device)

        # Predict
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        y_pred = teacher_model(x)

        # Compute loss (foward propagation)
        loss = criterion(y_pred, y_gt)
        
        # Compute gradients (backward propagation)
        loss.backward()
        
        # Update parameters (SGD)
        optimizer.step()

        # Clip weight
        max_norm = 15.0
        named_parameters = dict(teacher_model.named_parameters())
        for layer_name in ['layer1', 'layer2', 'layer3']:
            with torch.no_grad():
                weight = named_parameters['{}.weight'.format(layer_name)]
                bias = named_parameters['{}.bias'.format(layer_name)].unsqueeze(1)
                weight_bias = torch.cat((weight, bias),dim=1)
                norm = torch.norm(weight_bias, dim=1, keepdim=True).add_(1e-6)
                clip_coef = norm.reciprocal_().mul_(max_norm).clamp_(max=1.0)
                weight.mul_(clip_coef)
                bias.mul_(clip_coef)
                
        total_loss += loss.item()
        if step_count % 100 == 0:
            print('progress: {}\t/ {}\tloss: {}'.format(step_count, len(train_loader), loss.item()))

    list_lr_momentum_scheduler.step()
    print('loss: {}'.format(total_loss / len(train_loader)))

    ## Save model
    if epoch_count % 100 == 0:
        torch.save(teacher_model.state_dict(), './data/teacher-{}.pth'.format(epoch_count))


## Save model
torch.save(teacher_model.state_dict(), './data/teacher.pth')
