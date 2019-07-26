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
parser.add_argument('file_loc', action='store',type=str, default = "./flowers/valid/1/image_06739.jpg",help='Location of the image')
parser.add_argument('checkpoint', action="store", type=str, default ="checkpoint.pth",help='Load the trained model')
parser.add_argument('--gpu', action='store_true')
parser.add_argument('--top_k', action='store',type = int, default = 5, dest='top_k', help='Top K most likely classes')
parser.add_argument('--category_names', dest = "category_names",action="store", type=str, default ="cat_to_name.json",help='JSON file that maps the class values to other category names') 
                   
user_inputs = parser.parse_args()
file_loc = user_inputs.file_loc
chkpt = user_inputs.checkpoint
top_k = user_inputs.top_k
cat_names = user_inputs.category_names

gpu = False
args, _ = parser.parse_known_args()
if args.gpu:
    gpu = args.gpu
    
def load_checkpoint(chkpt):
    checkpoint = torch.load(chkpt)
    model = getattr(torchvision.models, checkpoint['arch'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.epochs = checkpoint['epochs']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    #optimizer.load_state_dict(checkpoint['optimizer'])
    return model


def process_image(file_loc):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    img = Image.open(file_loc)
    # Resize the images where the shortest side is 256 pixels, keeping the aspect ratio
    img_width, img_height = img.size
    aspect_ratio = img_width / img_height
    if aspect_ratio > 1:
        img = img.resize((round(aspect_ratio * 256), 256))
    else:
        img = img.resize((256, round(256 / aspect_ratio)))
    
    
    old_width, old_height = img.size
    new_width = 224
    new_height = 224
    left = (old_width - new_width)/2
    top = (old_height - new_height)/2
    right = (old_width + new_width)/2
    bottom = (old_height + new_height)/2
    img_crop = img.crop((round(left), round(top), round(right), round(bottom)))
    
    np_image = np.array(img_crop)
    
    np_image_0 = np.array(np_image) / 255

    np_image_norm = (np_image_0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    np_image_out =  np_image_norm.transpose((2, 0, 1))
            
    return np_image_out


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
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

def predict(image_path, model, top_k,device_choice):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device_choice == True:
        print("Use GPU")
        device = 'cuda'
    else:
        print("Use CPU")
        device ='cpu'
    
    np_image = process_image(image_path)

    model.to(device)
    model.eval()
    
    with torch.no_grad():
        images = torch.from_numpy(np_image)
        images = images.unsqueeze(0)
        images = images.type(torch.FloatTensor)
        images = images.to(device) # Move input tensors to the GPU/CPU

        output = model.forward(images)
        ps = torch.exp(output) # get the class probabilities from log-softmax

        #top_p, top_class = ps.topk(5, dim=1)
        top_p = torch.topk(ps, top_k)[0].tolist()[0]
        top_class = torch.topk(ps, top_k)[1].tolist()[0]
        inv_map = {v: k for k, v in model.class_to_idx.items()}
        classes = [inv_map[int(index)] for index in top_class]
        
    return top_p, classes


def main():
    model = load_checkpoint(chkpt)
    pil_image = file_loc
    with open(cat_names, 'r') as f:
        cat_to_name = json.load(f)

    probs, classes = predict(pil_image, model, top_k,gpu)
    print("probs: ", probs) 
    print("classes: ", classes)

    label_index = classes[np.argmax(probs)]
    y_pos = np.arange(top_k)
    labels = [cat_to_name[class1] for class1 in classes]
    print(labels)
    #f, plots = plt.subplots(2,1)
    #image = Image.open(pil_image)
    #plots[0].imshow(image)
    #plots[0].axis('off')
    #plots[0].set_title(cat_to_name[label_index])

    #plots[1].barh(y_pos, probs, align='center', color='blue')
    #plots[1].set_yticks(y_pos)
    #plots[1].set_yticklabels(labels)
    #plots[1].invert_yaxis()

if __name__== "__main__":
    main()
        
