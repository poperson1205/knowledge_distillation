import matplotlib.pyplot as plt
import numpy as np
import models
import torch.utils.data
import torchvision
import os

BASE_DIR = './data'

def evaluate(model_type='teacher', prefix='teacher'):
    if model_type == 'teacher':
        model = models.Teacher()
    elif model_type == 'student':
        model = models.Student()
       
    model_paths = []
    for epoch_count in range(0, 3000, 100):
        model_paths.append(os.path.join(BASE_DIR, '{}-{}.pth'.format(prefix, epoch_count)))
    model_paths.append(os.path.join(BASE_DIR, '{}.pth'.format(prefix)))
    
    # Get test data
    dataset = torchvision.datasets.MNIST('../mnist', train=False, download=True,
                                                    transform=torchvision.transforms.Compose([
                                                        torchvision.transforms.ToTensor(), 
                                                        torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                                        ]))

    # Create data loader
    batch_size = 10000
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    error_counts = np.zeros(shape=len(model_paths), dtype=np.int32)
    for model_count, model_path in enumerate(model_paths):
        model.load_state_dict(torch.load(model_path))
        model.eval()
        model = model.cuda()

        error_count = 0
        for x, y_gt in data_loader:
            # Preprocess input
            x = x.flatten(start_dim=1, end_dim=-1)
            x = x.cuda()
            y_gt = y_gt.cuda()

            # Predict label
            y_pred = model(x)
            with torch.no_grad():
                y_pred = torch.argmax(y_pred, dim=1)
                error_count += batch_size - torch.eq(y_pred, y_gt).sum()
        
        error_counts[model_count] = error_count
        print('{}-{}: {}'.format(prefix, model_count, error_count))

    return error_counts

# Evaluate accuracy
error_counts_teacher = evaluate('teacher', 'teacher')
error_counts_student = evaluate('student', 'student')
error_counts_student_distill = evaluate('student', 'student-distill')

# Store as file
np.save('./data/error_counts_teacher.npy', error_counts_teacher)
np.save('./data/error_counts_student.npy', error_counts_student)
np.save('./data/error_counts_student_distill.npy', error_counts_student_distill)

# # Load from file
# error_counts_teacher = np.load('./data/error_counts_teacher.npy')
# error_counts_student = np.load('./data/error_counts_student.npy')
# error_counts_student_distill = np.load('./data/error_counts_student_distill.npy')

# Prepare to plot
fig, ax = plt.subplots()

# Plot error
ax.plot(range(0, 3001, 100), error_counts_teacher, label='teacher')
ax.plot(range(0, 3001, 100), error_counts_student, label='student')
ax.plot(range(0, 3001, 100), error_counts_student_distill, label='student with distillation')
ax.set_xlabel('number of epochs')
ax.set_ylabel('number of errors')
ax.set_title('Learning curve')
ax.legend()

# Show
plt.show()