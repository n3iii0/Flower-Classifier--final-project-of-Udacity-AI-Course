import numpy as np
import pandas as pd
import torch 
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from PIL import Image
import argparse
from project import build, train, validation, process_image


parse = argparse.ArgumentParser(description='Train.py')
#type in shell: python train.py ./flowers /train /valid vgg16 0.001 120 'cuda' 5
parse.add_argument('data_dir', type=str, action='store', default="./flowers")
parse.add_argument('training_dir', type=str, action='store', default="/train")
parse.add_argument('valid_dir', type=str, action='store', default="/valid")
parse.add_argument('modelname', type=str, action='store', default="vgg16")
parse.add_argument('learnrate', type=float, action='store', default=0.001)
parse.add_argument('hidden', type=int, action='store', default = 120)
parse.add_argument('device', type=str, action='store', default='cuda')
parse.add_argument('epochs', type=int, action="store", default=5)

args = parse.parse_args()

data_dir = args.data_dir
training = args.train_dir
valid = args.valid_dir
modelname = args.modelname
learnrate = args.learnrate
hidden = args.hidden
epochs = args.epochs
device = args.device

train_dir = data_dir + training
valid_dir = data_dir + valid
train_transforms = transforms.Compose([transforms.RandomRotation(36),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229,0.224,0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485,0.456,0.406],
                                                           [0.229,0.224,0.225])])

train_datasets = datasets.ImageFolder(train_dir, transform = train_transforms)
valid_datasets = datasets.ImageFolder(valid_dir,transform = test_transforms)
trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle = True)   
validloader = torch.utils.data.DataLoader(valid_datasets, batch_size=16, shuffle = True)


model, criterion, optimizer, modelname  = build(modelname, learnrate, hidden)
model = train(model, optimizer, criterion, trainloader, validloader, modelname, epochs, hidden, device)

print('The Model has been build and trained')