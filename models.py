import torch
import torch.nn as nn
import torch.nn.functional as F

class Teacher(nn.Module):
    def __init__(self):
        # python requires to call ancestor's initilizer manually!
        super(Teacher, self).__init__()
        self.layer1 = nn.Linear(784, 1200)
        self.layer2 = nn.Linear(1200, 1200)
        self.layer3 = nn.Linear(1200, 10)
        self.dropout_20 = nn.Dropout(0.2)
        self.dropout_50 = nn.Dropout(0.5)

    def forward(self, x):
        x = self.dropout_20(x)
        x = F.relu(self.layer1(x))
        x = self.dropout_50(x)
        x = F.relu(self.layer2(x))
        x = self.dropout_50(x)
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
