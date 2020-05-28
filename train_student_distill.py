import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import scheduler

from models import Teacher
from models import Student

class CrossEntropyLossForSoftTarget(nn.Module):
    def __init__(self, T=20):
        super(CrossEntropyLossForSoftTarget, self).__init__()
        self.T = T
        self.softmax = nn.Softmax(dim=-1)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
    def forward(self, y_pred, y_gt):
        y_pred_soft = y_pred.div(self.T)
        y_gt_soft = y_gt.div(self.T)
        return -(self.softmax(y_gt_soft)*self.logsoftmax(y_pred_soft)).mean().mul(self.T*self.T)

MNIST_DIR = '../mnist/'

# Create gpu device
device = torch.device('cuda')
print(device)

# Create model
teacher_model = Teacher()
teacher_model.load_state_dict(torch.load('./data/teacher.pth'))
teacher_model.eval()
student_model = Student()

# Transfer
teacher_model.to(device)
student_model.to(device)

# Define Loss
criterion = nn.CrossEntropyLoss(reduction='mean')
criterion_soft = CrossEntropyLossForSoftTarget()

# Define optimizer
optimizer = optim.SGD(student_model.parameters(), lr=0.1)

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
                                            torchvision.transforms.ToTensor(), # image to Tensor
                                            torchvision.transforms.Normalize((0.1307,), (0.3081,)) # image, label
                                            ]))

train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)

student_model.train()
for epoch_count in range(3000):
    print('epoch: {}'.format(epoch_count))

    ## Optimize parameters
    total_loss = 0.0
    for step_count, (x, y_gt) in enumerate(train_loader):
        # Initialize gradients with 0
        optimizer.zero_grad()

        # Transfer device
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = x.to(device)
        y_gt = y_gt.to(device)

        # Compute soft label
        y_soft = teacher_model(x)

        # Predict
        y_pred = student_model(x)

        # Compute loss (foward propagation)
        loss = criterion(y_pred, y_gt) + criterion_soft(y_pred, y_soft)
        # loss = criterion(y_pred, y_gt)
        # loss = criterion_soft(y_pred, y_soft)
        
        # Compute gradients (backward propagation)
        loss.backward()
        
        # Update parameters (SGD)
        optimizer.step()

        # Clip weight
        max_norm = 15.0
        named_parameters = dict(student_model.named_parameters())
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
        torch.save(student_model.state_dict(), './data/student-distill-{}.pth'.format(epoch_count))


## Save model
torch.save(student_model.state_dict(), './data/student-distill.pth')
