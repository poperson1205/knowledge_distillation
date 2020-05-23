import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

MNIST_DIR = '../mnist/'

# Define network
class Teacher(nn.Module):
    def __init__(self):
        # python requires to call ancestor's initilizer manually!
        super(Teacher, self).__init__()
        self.layer1 = nn.Linear(784, 1200)
        self.layer2 = nn.Linear(1200, 1200)
        self.layer3 = nn.Linear(1200, 10)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class Student(nn.Module):
    def __init__(self):
        # python requires to call ancestor's initilizer manually!
        super(Student, self).__init__()
        self.layer1 = nn.Linear(784, 800)
        self.layer2 = nn.Linear(800, 800)
        self.layer3 = nn.Linear(800, 10)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

teacher_model = Teacher()
student_model = Student()


# # Train teacher
# ## Define Loss
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(teacher_model.parameters(), lr=0.001, momentum=0.9)

# ## Load dataset
# train_data = torchvision.datasets.MNIST(MNIST_DIR, train=True, download=True,
#                                         transform=torchvision.transforms.Compose([
#                                             torchvision.transforms.ToTensor(), # image to Tensor
#                                             torchvision.transforms.Normalize((0.1307,), (0.3081,)) # image, label
#                                             ]))
# train_loader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True)

# for epoch_count in range(3):
#     print('epoch: {}'.format(epoch_count))

#     ## Optimize parameters
#     total_loss = 0.0
#     for step_count, (x, y_gt) in enumerate(train_loader):
#         # Initialize gradients with 0
#         optimizer.zero_grad()

#         # Predict
#         x = torch.flatten(x, start_dim=1, end_dim=-1)
#         y_pred = teacher_model(x)

#         # Compute loss (foward propagation)
#         loss = criterion(y_pred, y_gt)
        
#         # Compute gradients (backward propagation)
#         loss.backward()

#         # Update parameters (SGD)
#         optimizer.step()

#         total_loss += loss.item()
#         if step_count % 1000 == 0:
#             print('progress: {}\t/ {}\tloss: {}'.format(step_count, len(train_loader), loss.item()))
    
#     print('loss: {}'.format(total_loss / len(train_loader)))


# # Save model
# torch.save(teacher_model.state_dict(), './data/teacher.pth')


# # Test
# test_data = torchvision.datasets.MNIST(MNIST_DIR, train=False, download=True,
#                                         transform=torchvision.transforms.Compose([
#                                             torchvision.transforms.ToTensor(), # image to Tensor
#                                             torchvision.transforms.Normalize((0.1307,), (0.3081,)) # image, label
#                                             ]))
# test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)
# error_count = 0
# total_loss = 0.0
# total_count = 0
# for step_count, (x, y_gt) in enumerate(test_loader):
#     x = torch.flatten(x, start_dim=1, end_dim=-1)
#     y_pred = teacher_model(x)

#     # Compute loss
#     total_loss += criterion(y_pred, y_gt).item()
    
#     # Check error
#     y_pred_argmax = torch.argmax(y_pred)
#     if y_pred_argmax.item() != y_gt.item():
#         error_count += 1

#     total_count += 1
#     if step_count % 1000 == 0:
#         print('progress: {}\t/ {}'.format(step_count, len(test_loader)))

# print('Test loss: {}'.format(total_loss / float(total_count)))
# print('Test error: {} / {}'.format(error_count, total_count))


# Train student
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

for epoch_count in range(3):
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


# Save model
torch.save(student_model.state_dict(), './data/student.pth')


# Test
test_data = torchvision.datasets.MNIST(MNIST_DIR, train=False, download=True,
                                        transform=torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor(), # image to Tensor
                                            torchvision.transforms.Normalize((0.1307,), (0.3081,)) # image, label
                                            ]))
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)
error_count = 0
total_loss = 0.0
total_count = 0
for step_count, (x, y_gt) in enumerate(test_loader):
    x = torch.flatten(x, start_dim=1, end_dim=-1)
    y_pred = student_model(x)

    # Compute loss
    total_loss += criterion(y_pred, y_gt).item()
    
    # Check error
    y_pred_argmax = torch.argmax(y_pred)
    if y_pred_argmax.item() != y_gt.item():
        error_count += 1

    total_count += 1
    if step_count % 1000 == 0:
        print('progress: {}\t/ {}'.format(step_count, len(test_loader)))

print('Test loss: {}'.format(total_loss / float(total_count)))
print('Test error: {} / {}'.format(error_count, total_count))


