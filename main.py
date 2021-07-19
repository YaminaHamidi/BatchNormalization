import matplotlib.pyplot as plt
import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets 
from torch.utils.data import DataLoader

from model import *
from helper_functions import * 

transform = transforms.Compose([transforms.Pad(2), 
                                transforms.ToTensor()])

trainData = datasets.MNIST(root = 'data',train = True,transform= transform , download = True)
testData = datasets.MNIST(root = 'data', train = False, transform=transform  )

batch_size = 128
trainLoader = torch.utils.data.DataLoader(trainData, batch_size, shuffle=True)
testLoader = torch.utils.data.DataLoader(testData, batch_size, shuffle=True)


### LeNet (witout Batch Normalization)
model = LeNet5()
n_epochs = 10
lr = 0.9 
model,train_loss_overTime, test_loss_overTime,train_class_correct, test_class_correct,train_class_total, test_class_total = train(model, n_epochs, lr, trainLoader, testLoader )

print_train_class_accuracy(model,train_class_correct,train_class_total)
print_test_class_accuracy(model,test_class_correct,test_class_total)
plot_loss(train_loss_overTime, test_loss_overTime)
plot_classif_samples(model, testLoader)


### LeNet (with Batch Normalization)
model = LeNet5_batchNorm()
n_epochs = 10
lr = 0.9 
model,train_loss_overTime, test_loss_overTime,train_class_correct, test_class_correct,train_class_total, test_class_total = train(model, n_epochs, lr, trainLoader, testLoader )

print_train_class_accuracy(model,train_class_correct,train_class_total)
print_test_class_accuracy(model,test_class_correct,test_class_total)
plot_loss(train_loss_overTime, test_loss_overTime)
plot_classif_samples(model, testLoader)



