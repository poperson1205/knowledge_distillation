# Knowledge Distillation

This repository is for testing knowledge distillation. Especially for experiments on MNIST, which is described in this paper [1]: Hinton et. al. "Distilling the Knowledge in a Neural Network". NIPS2014. Details of model structures and training hyper-paremters are written in another paper [2]: Hinton et. al. "Improving neural networks by preventing co-adaption of feature detectors". https://arxiv.org/abs/1207.0580

I also wrote a story in Medium: [Knowledge Distillation for Object Detection 1: Start from simple classification model](https://medium.com/@poperson1205/knowledge-distillation-for-object-detection-1-start-from-simple-classification-model-921e1b2bfed2)

**scheduler.py is copied from timesler's repository (https://github.com/timesler/lr-momentum-scheduler)**

## Prerequisites
- Pytorch (1.5)
- Numpy
- matplotlib

## Quick Start

### Pull docker container

```bash
docker pull poperson1205/knowledge_distillation
```

### Train teacher
```bash
python train_teacher.py
```

### Train student
```bash
python train_student.py
```

### Train student with knowledge distillation
```bash
python train_student_distill.py
```

### Evaluate and visualize trained networks
```bash
python evaluate.py
```

## Result

- Teacher: 100 / 10000 (1.00%)
- Student: 171 / 10000 (1.71%)
- Student with KD: 111 / 10000 (1.11%)

![](learning_curve.png)

Although the accuracy of teacher model (100 errors) is not good as written in the original paper (74 errors), we could see the power of the knowledge distillation by comparing vanilla student model (171 errors) and distilled student model (111 errors).

## Reference
[1] Hinton et. al. "Distilling the Knowledge in a Neural Network". NIPS2014.

[2] Hinton et. al. "Improving neural networks by preventing co-adaption of feature detectors". https://arxiv.org/abs/1207.0580

[3] Presentation material of paper[1]: https://www.ttic.edu/dl/dark14.pdf