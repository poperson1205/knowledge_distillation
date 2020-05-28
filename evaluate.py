import argparse

import torch
import torchvision

from models import Teacher
from models import Student

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('type', help='Type of network, which should be either "Teacher" or "Student"')
    parser.add_argument('path', help='File path for the target network')
    args = parser.parse_args()

    # Create model
    model = None
    if args.type == 'Teacher':
        model = Teacher()
    elif args.type == 'Student':
        model = Student()

    # Load state_dict
    model.load_state_dict(torch.load(args.path))
    model.eval()

    test_data = torchvision.datasets.MNIST('../mnist', train=False, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))]))
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    error_count = 0
    total_count = 0
    for step_index, (x, y_gt) in enumerate(test_loader):
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        y_pred = model(x)
        
        y_pred_argmax = torch.argmax(y_pred).item()
        if y_pred_argmax != y_gt:
            error_count += 1
        
        total_count += 1

        if step_index % 1000 == 0:
            print('progress: {}\t/ {}'.format(step_index, len(test_loader)))

    print('error: {}\t/ {}\t-->\t{}(%)'.format(error_count, total_count, float(error_count)/float(total_count)*100.0))