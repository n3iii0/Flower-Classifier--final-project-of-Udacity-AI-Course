#!/usr/bin/env python
# coding: utf-8

# # Developing an AI application
# 
# Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications. 
# 
# In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories, you can see a few examples below. 
# 
# <img src='assets/Flowers.png' width=500px>
# 
# The project is broken down into multiple steps:
# 
# * Load and preprocess the image dataset
# * Train the image classifier on your dataset
# * Use the trained classifier to predict image content
# 
# We'll lead you through each part which you'll implement in Python.
# 
# When you've completed this project, you'll have an application that can be trained on any set of labeled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. Go build your own dataset and make something new.
# 
# First up is importing the packages you'll need. It's good practice to keep all the imports at the beginning of your code. As you work through this notebook and find you need to import a package, make sure to add the import up here.
# 
# Please make sure if you are running this notebook in the workspace that you have chosen GPU rather than CPU mode.

# In[1]:


#get_ipython().run_line_magic('matplotlib', 'inline')
#get_ipython().run_line_magic('config', "InlineBackend.figure_format ='retina'")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch 
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from PIL import Image


# ## Load the data
# 
# Here you'll use `torchvision` to load the data ([documentation](http://pytorch.org/docs/0.3.0/torchvision/index.html)). The data should be included alongside this notebook, otherwise you can [download it here](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz). The dataset is split into three parts, training, validation, and testing. For the training, you'll want to apply transformations such as random scaling, cropping, and flipping. This will help the network generalize leading to better performance. You'll also need to make sure the input data is resized to 224x224 pixels as required by the pre-trained networks.
# 
# The validation and testing sets are used to measure the model's performance on data it hasn't seen yet. For this you don't want any scaling or rotation transformations, but you'll need to resize then crop the images to the appropriate size.
# 
# The pre-trained networks you'll use were trained on the ImageNet dataset where each color channel was normalized separately. For all three sets you'll need to normalize the means and standard deviations of the images to what the network expects. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`, calculated from the ImageNet images.  These values will shift each color channel to be centered at 0 and range from -1 to 1.
#  

# In[2]:


data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


# In[3]:

'''
# TODO: Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(30),
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

# TODO: Load the datasets with ImageFolder
train_datasets = datasets.ImageFolder(train_dir, transform = train_transforms)
test_datasets = datasets.ImageFolder(test_dir, transform = test_transforms)
valid_datasets = datasets.ImageFolder(valid_dir,transform = test_transforms)
# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle = True)
testloader = torch.utils.data.DataLoader(test_datasets, batch_size=32, shuffle = True)
validloader = torch.utils.data.DataLoader(valid_datasets, batch_size=16, shuffle = True)
'''

# ### Label mapping
# 
# You'll also need to load in a mapping from category label to category name. You can find this in the file `cat_to_name.json`. It's a JSON object which you can read in with the [`json` module](https://docs.python.org/2/library/json.html). This will give you a dictionary mapping the integer encoded categories to the actual names of the flowers.

# In[4]:


import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


# # Building and training the classifier
# 
# Now that the data is ready, it's time to build and train the classifier. As usual, you should use one of the pretrained models from `torchvision.models` to get the image features. Build and train a new feed-forward classifier using those features.
# 
# We're going to leave this part up to you. Refer to [the rubric](https://review.udacity.com/#!/rubrics/1663/view) for guidance on successfully completing this section. Things you'll need to do:
# 
# * Load a [pre-trained network](http://pytorch.org/docs/master/torchvision/models.html) (If you need a starting point, the VGG networks work great and are straightforward to use)
# * Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
# * Train the classifier layers using backpropagation using the pre-trained network to get the features
# * Track the loss and accuracy on the validation set to determine the best hyperparameters
# 
# We've left a cell open for you below, but use as many as you need. Our advice is to break the problem up into smaller parts you can run separately. Check that each part is doing what you expect, then move on to the next. You'll likely find that as you work through each part, you'll need to go back and modify your previous code. This is totally normal!
# 
# When training make sure you're updating only the weights of the feed-forward network. You should be able to get the validation accuracy above 70% if you build everything right. Make sure to try different hyperparameters (learning rate, units in the classifier, epochs, etc) to find the best model. Save those hyperparameters to use as default values in the next part of the project.
# 
# One last important tip if you're using the workspace to run your code: To avoid having your workspace disconnect during the long-running tasks in this notebook, please read in the earlier page in this lesson called Intro to GPU Workspaces about Keeping Your Session Active. You'll want to include code from the workspace_utils.py module.

# In[5]:


# TODO: Do validation on the test set
def validation(model,validloader,criterion, device):
    test_loss = 0
    accuracy = 0
    for images, labels in validloader:
        images, labels = images.to(device), labels.to(device)
        #images.resize_(images.shape[0], 784)
        model.to(device)
        output = model.forward(images)
        test_loss += criterion(output, labels).item()
        ps = torch.exp(output)
        equality = labels.data == ps.max(dim = 1)[1]
        accuracy += equality.type(torch.FloatTensor).mean()
    return test_loss, accuracy 


# In[6]:


def build(modelname = 'vgg16', learnrate = 0.001, hidden=120):
    
    if modelname == 'vgg16':
        model = models.vgg16(pretrained = True)
        input_size = 25088
    elif modelname == 'densenet121':
        model = models.densenet121(pretrained = True)
        input_size = 1024
    elif modelname == "resnet50":
        model = models.resnet50(pretrained = True)
        input_size = 152
    else:
        print('No valid structure, please type in "vgg16", "densenet121" or "resnet50"')
    
    for param in model.parameters():
        param.requires_grad = False

    hidden_size = [hidden,100,82]
    output_size = 102

    classifier = nn.Sequential(nn.Dropout(p=0.5),
                          nn.Linear(input_size, hidden_size[0]),
                          nn.ReLU(),
                          nn.Linear(hidden_size[0], hidden_size[1]),
                          nn.ReLU(),
                          nn.Linear(hidden_size[1],hidden_size[2]),
                          nn.ReLU(),
                          nn.Linear(hidden_size[2],output_size),
                          nn.LogSoftmax(dim=1))

    model.classifier = classifier
    criterion = nn.NLLLoss()
    #criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = learnrate)
    
    return model, criterion, optimizer, modelname 


def train(model, optimizer, criterion,trainloader, validloader,  modelname='vgg16', epochs = 5, hidden = 120, device = 'cuda'):
    steps = 0
    print_every = 5
    
    if device == 'cuda' and torch.cuda.is_available():
        print('Training on GPU')
        device = 'cuda'
    else:
        print('Training on CPU')
        device = 'cpu'
    
    model.to(device)
    for e in range(epochs):
        model.train()
        losscount = 0
        for images, labels in trainloader:
            steps +=1
            #images.resize_(images.shape()[0], 3, 784)
            images,labels  = images.to(device), labels.to(device)
        
            #images.resize_(32,784)

            optimizer.zero_grad()

            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        
            losscount += loss.item()
            if steps % print_every == 0:
                model.eval()
                with torch.no_grad():
                    testloss, accuracy = validation(model, validloader, criterion, device)
                    print("Epoch: {}/{}.. ".format(e+1, epochs),
                          "Training Loss: {:.3f}.. ".format(losscount/print_every),
                          "Validation Loss: {:.3f}.. ".format(testloss/len(validloader)),
                          "Validation Accuracy: {:.3f}".format(accuracy/len(validloader)))
                    losscount = 0
    
    model.train()
    
    model.to('cpu')
    model.class_to_idx = train_datasets.class_to_idx
    checkpoint = {'model': modelname,
                  'model_state_dict': model.state_dict(),
                  'class_to_idx' : model.class_to_idx,
                  'opt_state_dict':optimizer.state_dict(),
                  'epochs': epochs}
    torch.save(checkpoint, 'checkpoint.pth')
    
    return model


# ## Testing your network
# 
# It's good practice to test your trained network on test data, images the network has never seen either in training or validation. This will give you a good estimate for the model's performance on completely new images. Run the test images through the network and measure the accuracy, the same way you did validation. You should be able to reach around 70% accuracy on the test set if the model has been trained well.

# In[7]:


def testing(model, criterion, device = 'cuda'):
    if device == 'cuda' and torch.cuda.is_available():
        print('Testing on GPU')
        device = 'cuda'
    else:
        print('Testing on CPU')
        device = 'cpu'
    
    model.to(device)
    model.eval()
    testloss = 0
    accuracy = 0
    for images, labels in testloader:
        images, labels = images.to('cuda'), labels.to('cuda')
        #images.resize_(images.shape[0], 784)
        images, labels = images.to(device), labels.to(device)
        output = model.forward(images)
        testloss += criterion(output, labels)

        ps = torch.exp(output)
        equality = labels.data == ps.max(dim = 1)[1]
        accuracy += equality.type(torch.FloatTensor).mean()
    
    print("Test Loss: {:.3f}.. ".format(testloss/len(testloader)))
    print("Testimage Accuracy: {:.3f}".format((accuracy / len(testloader))*100))


# ## Save the checkpoint
# 
# Now that your network is trained, save the model so you can load it later for making predictions. You probably want to save other things such as the mapping of classes to indices which you get from one of the image datasets: `image_datasets['train'].class_to_idx`. You can attach this to the model as an attribute which makes inference easier later on.
# 
# ```model.class_to_idx = image_datasets['train'].class_to_idx```
# 
# Remember that you'll want to completely rebuild the model later so you can use it for inference. Make sure to include any information you need in the checkpoint. If you want to load the model and keep training, you'll want to save the number of epochs as well as the optimizer state, `optimizer.state_dict`. You'll likely want to use this trained model in the next part of the project, so best to save it now.

# In[8]:


#in function train


# ## Loading the checkpoint
# 
# At this point it's good to write a function that can load a checkpoint and rebuild the model. That way you can come back to this project and keep working on it without having to retrain the network.

# In[9]:


def loadcheckpoint(filepath, train = False):
    checkpoint = torch.load(filepath)
    model, criterion, optimizer, modelname  = build(checkpoint['model'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['model_state_dict'])
    if train == True:
        train(model, optimizer, criterion, checkpoint['model'], checkpoint['epochs'])
    
    return(model)


# # Inference for classification
# 
# Now you'll write a function to use a trained network for inference. That is, you'll pass an image into the network and predict the class of the flower in the image. Write a function called `predict` that takes an image and a model, then returns the top $K$ most likely classes along with the probabilities. It should look like 
# 
# ```python
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```
# 
# First you'll need to handle processing the input image such that it can be used in your network. 
# 
# ## Image Preprocessing
# 
# You'll want to use `PIL` to load the image ([documentation](https://pillow.readthedocs.io/en/latest/reference/Image.html)). It's best to write a function that preprocesses the image so it can be used as input for the model. This function should process the images in the same manner used for training. 
# 
# First, resize the images where the shortest side is 256 pixels, keeping the aspect ratio. This can be done with the [`thumbnail`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) or [`resize`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) methods. Then you'll need to crop out the center 224x224 portion of the image.
# 
# Color channels of images are typically encoded as integers 0-255, but the model expected floats 0-1. You'll need to convert the values. It's easiest with a Numpy array, which you can get from a PIL image like so `np_image = np.array(pil_image)`.
# 
# As before, the network expects the images to be normalized in a specific way. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`. You'll want to subtract the means from each color channel, then divide by the standard deviation. 
# 
# And finally, PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array. You can reorder dimensions using [`ndarray.transpose`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.transpose.html). The color channel needs to be first and retain the order of the other two dimensions.

# In[10]:


def process_image(image):
    pil_image = Image.open(image)
    #pil_image = pil_image.resize((256,256))
    size = 256
    width,height = pil_image.size
    shortest_side = min(width,height)
    resized = pil_image.resize((int((width/shortest_side)*size), int((height/shortest_side)*size)))
    d1 = (256-224)/2
    d2 = (256+224)/2
    cropped = resized.crop((d1,d1,d2,d2))
    #transform = transforms.Compose([transforms.Resize(256),
                                    #transforms.CenterCrop(224),
                                    #transforms.ToTensor(),
                                    #transforms.Normalize([0.485, 0.456, 0.406],
                                                            #[0.229,0.224,0.225])])

    #pil_tensor = transform(pil_image).float()
    
    np_image = np.array(cropped) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    np_image = (np_image - mean)/std
    np_image = np.ndarray.transpose(np_image,(2,0,1))
    
    return np_image


# To check your work, the function below converts a PyTorch tensor and displays it in the notebook. If your `process_image` function works, running the output through this function should return the original image (except for the cropped out portions).

# In[11]:


def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


# ## Class Prediction
# 
# Once you can get images in the correct format, it's time to write a function for making predictions with your model. A common practice is to predict the top 5 or so (usually called top-$K$) most probable classes. You'll want to calculate the class probabilities then find the $K$ largest values.
# 
# To get the top $K$ largest values in a tensor use [`x.topk(k)`](http://pytorch.org/docs/master/torch.html#torch.topk). This method returns both the highest `k` probabilities and the indices of those probabilities corresponding to the classes. You need to convert from these indices to the actual class labels using `class_to_idx` which hopefully you added to the model or from an `ImageFolder` you used to load the data ([see here](#Save-the-checkpoint)). Make sure to invert the dictionary so you get a mapping from index to class as well.
# 
# Again, this method should take a path to an image and a model checkpoint, then return the probabilities and classes.
# 
# ```python
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```

# In[12]:


def predict(image_path, model, topk=5, device = 'cuda'):
    
    if device == 'cuda' and torch.cuda.is_available():
        print('Prediction calculated on GPU')
        device = 'cuda'
    else:
        print('Prediction calculated on CPU')
        device = 'cpu'
    
    model.to(device)
    model.eval()
    image = torch.FloatTensor(process_image(image_path))
    
    index_to_class = {val: key for key, val in model.class_to_idx.items()}
    #top_classes = [index_to_class[each] for each in indexes]
    
    image.unsqueeze_(0) 
    #print(image.shape)
    image = image.to(device)
    
    with torch.no_grad():
        forward = model.forward(image)
    
    #results uses LogSoftmax. to convert to normal percantages use torch.exp
    class_to_idx = model.class_to_idx
    
    
    results = torch.exp(forward
                       )
    prob, index = results.topk(topk)
    
    prob = prob.to('cpu')
    index = index.to('cpu')
    
    prob = prob.numpy()[0]
    index = index.numpy()[0]
    
    mapped_index=[]
    prob_topk=[]
    
    for i, idx in enumerate(index):
        mapped_index.append(index_to_class[idx])
        prob_topk.append(prob[i])
        
        #print(mapped_index)
        #print(prob_topk)
    '''cat_topk=[]
    prob_topk=[]
    
    for i in range(topk):
        for key, value in class_to_idx.items():
            if value == index[i]:
                cat_topk.append(cat_to_name[key])
                prob_topk.append(prob[i])
                #print("Categories:{}___Probabilites:{}".format(cat_topk[i],prob_topk[i]))'''
                
    
    return prob_topk, mapped_index
    
    
    


# ## Sanity Checking
# 
# Now that you can use a trained model for predictions, check to make sure it makes sense. Even if the testing accuracy is high, it's always good to check that there aren't obvious bugs. Use `matplotlib` to plot the probabilities for the top 5 classes as a bar graph, along with the input image. It should look like this:
# 
# <img src='assets/inference_example.png' width=300px>
# 
# You can convert from the class integer encoding to actual flower names with the `cat_to_name.json` file (should have been loaded earlier in the notebook). To show a PyTorch tensor as an image, use the `imshow` function defined above.

# In[13]:


'''
model, criterion, optimizer, modelname  = build(modelname = 'densenet121')
train(model, optimizer, criterion, modelname, epochs = 10)
model = loadcheckpoint('checkpoint.pth')
criterion = nn.NLLLoss()
testing(model, criterion, device = 'cuda')
image_path = test_dir + '/71/image_04482.jpg'
prob, cat = predict(image_path, model, 5)

#prob = prob.to('cpu')
#number = number.to('cpu')
image = process_image(image_path)
#imshow(image)



flowers = []

for e in range(0,len(cat)):
    flowers.append(cat_to_name[str(cat[e])])


img= Image.open(image_path)
fig, (ax1, ax2) = plt.subplots(figsize=(6,10), ncols=1, nrows=2)
ax1.set_title(cat_to_name[str(cat[0])])
ax1.imshow(img)
ax1.axis('off')


y_pos = np.arange(len(flowers))
ax2.barh(y_pos, prob, align ='center', color = 'blue')
ax2.set_xlabel('Probabilities')
ax2.set_yticks(y_pos)
ax2.set_yticklabels(flowers)
ax2.invert_yaxis()'''


