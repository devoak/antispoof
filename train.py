import cv2
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import sampler
import torchvision
import torchvision.transforms as T
import torch.nn.functional as F
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.models as models

import warnings
warnings.filterwarnings("ignore")



#TODO - extract all parameters to a-lya config file
NUM_EPOCHS = 100
BATCH_SIZE = 32
PRINT_EVERY = 100
CHECKING_BATCH_AMOUNT = 5
USE_GPU = True
DTYPE = torch.float32
if USE_GPU and torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
print('using device:', DEVICE)



def train(model, optimizer, trainLoaders, valLoaders, num_epochs = 1):
    model = model.to(device = DEVICE)
    loaderIndex = 0
    xTensorsToConcat = []
    yTensorsToConcat = []
    for e in range(num_epochs):
        print('epoch','(',e,')')
        for t, (x, _) in enumerate(trainLoaders[loaderIndex % 2]):
            if x.size(0) != BATCH_SIZE:
                break
            loaderIndex += 1
            y = torch.zeros(BATCH_SIZE) if loaderIndex % 2 == 0 else torch.ones(BATCH_SIZE)
            xTensorsToConcat.append(x)
            yTensorsToConcat.append(y)
            if loaderIndex == 2:
                loaderIndex = 0
                randomIndicies = torch.randperm(BATCH_SIZE * 2)
                x = torch.cat(xTensorsToConcat)[randomIndicies]
                y = torch.cat(yTensorsToConcat)[randomIndicies]
                x = x.to(device = DEVICE, dtype = DTYPE)
                y = y.to(device = DEVICE, dtype = torch.long)
                scores = model(x)
                loss = F.cross_entropy(scores, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if t % PRINT_EVERY == 0:
                    print('Iteration %d, loss = %.4f' % (t, loss.item()))
                    print('Validation Accuracy')
                    checkAccuracy(valLoaders, model)
                    print('Train Accuracy')
                    checkAccuracy(trainLoaders, model)
                    print()


def checkAccuracy(loaders, model):
    print("checkAccuracy")
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for loaderIndex in range(len(loaders)):
            y = torch.zeros(BATCH_SIZE) if loaderIndex == 0 else torch.ones(BATCH_SIZE)
            for t, (x, _) in enumerate(loaders[loaderIndex]):
                if t > CHECKING_BATCH_AMOUNT:
                    break
                x = x.to(device = DEVICE, dtype = DTYPE)  # move to device, e.g. GPU
                y = y.to(device = DEVICE, dtype=torch.long)
                scores = model(x)
                _, argMaxIndicies = scores.max(1)
                num_samples += argMaxIndicies.size(0)
                num_correct += (argMaxIndicies == y).sum()
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))


def _main():
    transform = T.Compose([T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    datasetTrainReal = torchvision.datasets.ImageFolder('./datasets/trainReal', transform = transform)
    datasetTrainSpoof = torchvision.datasets.ImageFolder('./datasets/trainSpoof', transform = transform)
    datasetValReal = torchvision.datasets.ImageFolder('./datasets/valReal', transform = transform)
    datasetValSpoof = torchvision.datasets.ImageFolder('./datasets/valSpoof', transform = transform)
    
    NUM_TRAIN_REAL = 1223
    NUM_TRAIN_SPOOF = 7076
    NUM_VAL_REAL = 373
    NUM_VAL_SPOOF = 632
    
    loaderTrainReal = torch.utils.data.DataLoader(datasetTrainReal, batch_size = BATCH_SIZE,
                                                  sampler = sampler.SubsetRandomSampler(range(NUM_TRAIN_REAL)))
    loaderTrainSpoof = torch.utils.data.DataLoader(datasetTrainSpoof, batch_size = BATCH_SIZE,
                                                   sampler = sampler.SubsetRandomSampler(range(NUM_TRAIN_SPOOF)))
    loaderValReal = torch.utils.data.DataLoader(datasetValReal, batch_size = BATCH_SIZE,
                                                sampler = sampler.SubsetRandomSampler(range(NUM_VAL_REAL)))
    loaderValSpoof = torch.utils.data.DataLoader(datasetValSpoof, batch_size = BATCH_SIZE,
                                                 sampler = sampler.SubsetRandomSampler(range(NUM_VAL_SPOOF)))


    learning_rate = 1e-4
    model = models.squeezenet1_0(num_classes = 2)
    optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum=0.9, nesterov=True)

    train(model, optimizer, [loaderTrainReal, loaderTrainSpoof], [loaderValReal, loaderValSpoof], num_epochs = NUM_EPOCHS)
    model.save_state_dict('model.pt')

if __name__ == '__main__':
    _main()














