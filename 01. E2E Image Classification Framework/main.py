import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
# from torchsummary import summary

import torchvision

import os
import argparse
from tqdm import tqdm

from models import *
from utils import *

import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

model_name = "custom_animal"
# CUSTOM, CIFAR10
data_name = "CUSTOM"

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--lr', '-lr', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--b', '-b', default=128, type=int, help='batch size')  
parser.add_argument('--e', '-e', default=5, type=int, help='no of epochs') 
parser.add_argument('--norm', default="batch", type=str, help='Normalization Type')
parser.add_argument('--name', default=model_name, type=str, help='Name of pth model file to be stored')   
parser.add_argument('--n', '-n', default=10, type=int, help='No of Images to be displayed after prediction (should be multiple of 5)') 
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')                  
args = parser.parse_args()

def main(model_name = args.name, lr = args.lr, batch_size = args.b, epochs = args.e, norm = args.norm, n = args.n, resume = args.resume):
# def main(lr = 0.007, batch_size = 128, epochs = 5, norm = "batch", n = 10, resume = False):
    global data_name
    SEED = 69
    use_cuda = torch.cuda.is_available()

    torch.manual_seed(SEED)
    if use_cuda:
        torch.cuda.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n****************************************************************************\n")
    print(f"Device Used: {device}\n")
    print("\n****************************************************************************\n")
    
    model = train_model(data_name, model_name, resume, device, norm, epochs = epochs, batch_size = batch_size, learning_rate = lr)

    print("\n****************************************************************************\n")

    print("*****Correctly Classified Images*****\n")

    image_prediction(data_name, model, "Correctly Classified Images", n=n,r=int(n/5),c=5, misclassified = False)

    # print("\n****************************************************************************\n")

    # print("*****Correctly Classified GradCam Images*****\n")

    # image_prediction("CIFAR10", model, "Correctly Classified GradCam Images", n=n,r=int(n/5),c=5, misclassified = False, gradcam=True)

    print("\n****************************************************************************\n")

    print("*****Misclassified Images*****\n")

    image_prediction(data_name, model, "Misclassified Images", n=n,r=int(n/5),c=5, misclassified = True)

    # print("\n****************************************************************************\n")

    # print("*****Misclassified GradCam Images*****\n")

    # image_prediction("CIFAR10", model, "Misclassified GradCam Images", n=n,r=int(n/5),c=5, misclassified = True, gradcam=True)

    # print("\n****************************************************************************\n")

if __name__ == "__main__":
    main()