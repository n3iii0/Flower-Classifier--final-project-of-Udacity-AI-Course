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
import json

from project import predict, loadcheckpoint

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
parse = argparse.ArgumentParser(description='Predict.py')
#type in shell python predict.py ./flowers /71/image_04482.jpg checkpoint.pth 5 'cuda'
parse.add_argument('data_dir', type=str, action='store', default="./flowers")
parse.add_argument('image_source', type=str, action='store', default='/test/71/image_04482.jpg')
parse.add_argument('checkpoint_path', type=str, action='store', default='checkpoint.pth')
parse.add_argument('topk', type=int, action='store', default=5)
#parse.add_argument('totrain', type=str, action='store', default='yes')
parse.add_argument('device', type=str, action='store', default='cuda')


args = parse.parse_args()

data_dir = args.data_dir
image_source = args.image_source
checkpoint_path = args.checkpoint_path
topk = args.topk
device = args.device
#totrain = args.totrain

image_path = data_dir + image_source
model = loadcheckpoint(checkpoint_path)
    
prob,cat = predict(image_path, model, topk, device) 
    
flowers = []
    
for e in range(0,len(cat)):
    
    flowers.append(cat_to_name[str(cat[e])])
    #print(flowers[e])
    print('The image might be a {} with a percantage of {}'.format(flowers[e],prob[e]*100))
    #print(probabilities[e])
    
    
    
    