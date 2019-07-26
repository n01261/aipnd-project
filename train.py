import torchvision
from torchvision import datasets, transforms, models
import json
import torch
from torch import nn
from torch import optim
from collections import OrderedDict
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('data_dir', action='store',type=str, default = "flowers",help='Main folder of dataset')
parser.add_argument('--gpu', action='store_true')
parser.add_argument('--epochs', action="store", dest = "epochs", type=int, default = 8, help='Number of epochs to train')
parser.add_argument('--arch', action="store", dest =  "arch",type=str, default = "vgg19",help='Model architecture to use (vgg19 or densenet121)')
parser.add_argument('--learning_rate', dest = "learning_rate",action="store", type=float, default = 0.01, help='Learning rate of the optimizer')
parser.add_argument('--hidden_units', dest = "hidden_units",action="store", type=int, default = 1024, help='Number of hidden units for the model')
parser.add_argument('--checkpoint', dest = "checkpoint",action="store", type=str, default ="checkpoint.pth",help='Save the trained model')
                   
user_inputs = parser.parse_args()
dir_loc = user_inputs.data_dir
chckpt = user_inputs.checkpoint
lr = user_inputs.learning_rate
architecture = user_inputs.arch
hidden_units = user_inputs.hidden_units
epochs = user_inputs.epochs


gpu = False
args, _ = parser.parse_known_args()
if args.gpu:
    gpu = args.gpu



def load_data(dir_loc):
    data_dir =  dir_loc
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # TODO: Define your transforms for the training, validation, and testing sets
    data_transforms_train = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485,0.456,0.406],[0.229, 0.224, 0.225])])

    data_transforms_validation = transforms.Compose([transforms.Resize(255),
                                                    transforms.CenterCrop(244),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485,0.456,0.406],[0.229, 0.224, 0.225])])
    
    data_transforms_test = transforms.Compose([transforms.Resize(255),
                                                    transforms.CenterCrop(244),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485,0.456,0.406],[0.229, 0.224, 0.225])])


    image_train = datasets.ImageFolder(train_dir, transform=data_transforms_train)
    image_validation = datasets.ImageFolder(valid_dir, transform=data_transforms_validation)
    image_test = datasets.ImageFolder(test_dir, transform=data_transforms_test)

    train_dataloaders = torch.utils.data.DataLoader(image_train, batch_size=32, shuffle=True)
    validation_dataloaders = torch.utils.data.DataLoader(image_validation, batch_size=32)
    test_dataloaders  = torch.utils.data.DataLoader(image_test, batch_size=32)

    image_datasets = [image_train, image_validation, image_test]
    dataloaders = [train_dataloaders, validation_dataloaders, test_dataloaders]
    
    return image_datasets, dataloaders

def build_model(architecture = 'vgg19', lr = 0.01, hidden_units = 1024):
    
    if architecture == 'vgg19':
        model = models.vgg19(pretrained=True)
        input_feature = 25088
    elif architecture == 'densenet121':
        model = models.densenet121(pretrained=True)
        input_feature = 1024
    for param in model.parameters():
        param.requires_grad = False
  
    classifier = nn.Sequential(OrderedDict([
                             ('fc1', nn.Linear(input_feature, hidden_units)),
                             ('drop', nn.Dropout(p=0.3)),
                             ('relu', nn.ReLU()),
                             ('fc2', nn.Linear(hidden_units, 102)),
                            ('output', nn.LogSoftmax(dim=1))
                             ]))
    
    model.classifier = classifier

    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.classifier.parameters(), lr)
    return model, criterion, optimizer
    
def train_model(device_choice , model, criterion, optimizer,epochs,image_datasets,dataloaders,architecture,lr,chckpt):
    if device_choice == True:
        print("Use GPU")
        device = 'cuda'
    else:
        print("Use CPU")
        device ='cpu'
    model.to(device);

    running_loss = 0

    for epoch in range(epochs):
        for inputs, labels in dataloaders[0]:
        # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
        
            optimizer.zero_grad()
        
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        
        validation_loss = 0
        accuracy = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in dataloaders[1]:
                inputs, labels = inputs.to(device), labels.to(device)
                logps = model.forward(inputs)
                batch_loss = criterion(logps, labels)
               
                validation_loss += batch_loss.item()
                    
                # Calculate accuracy
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
        print(f"Epoch {epoch+1}/{epochs}.. "
            f"Training loss: {running_loss/len(dataloaders[0]):.3f}.. "
            f"Validation loss: {validation_loss/len(dataloaders[1]):.3f}.. "
            f"Validation accuracy: {accuracy/len(dataloaders[1]):.3f}")
        running_loss = 0
        model.train()
        
        model.class_to_idx = image_datasets[0].class_to_idx
    
    if architecture == 'vgg19':
        input_feature = 25088
    elif architecture == 'densenet121':
        input_feature = 1024
    
    checkpoint = {'input_size': input_feature,
                  'output_size': 102,
                  'arch': architecture,
                  'learning_rate': lr,
                  'classifier' : model.classifier,
                  'epochs': epochs,
                  'optimizer': optimizer.state_dict(),
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx}




    torch.save(checkpoint, chckpt)
        

def main():
    
    image_datasets, dataloaders = load_data(dir_loc)
    model, criterion, optimizer = build_model(architecture, lr, hidden_units)
    train_model(gpu , model, criterion, optimizer,epochs,image_datasets,dataloaders,architecture,lr,chckpt)
    


if __name__== "__main__":
    main()
        

    
