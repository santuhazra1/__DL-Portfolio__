import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary

import os
from tqdm import tqdm
import numpy as np

from .dataloader import dataloaders
from .optimizer import get_optimizer
from .test import test
from .plot_graphs import plot_single
from .lr_sheduler import get_lr_sheduler

best_acc = 0
start_epoch = 0

def train(model, device, train_loader, optimizer, scheduler, criterion, l1_reg = None):
    model.train()

    # collect stats - for accuracy calculation
    correct = 0
    processed = 0
    epoch_loss = 0
    epoch_accuracy = 0
    pbar = tqdm(train_loader)

    for batch_id, (data, target) in enumerate(pbar):
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        # Gather prediction and calculate loss + backward pass + optimize weights
        label_pred = model(data)
        label_loss = criterion(label_pred, target)

        # L1 regularization
        if l1_reg is not None:
            l1_criterion = nn.L1Loss(size_average=False)
            l1_reg_loss = 0
            for param in model.parameters():
                l1_reg_loss += l1_criterion(param, torch.zeros_like(param))
                # print("L1 reg loss: ", l1_reg_loss)
            label_loss += l1_reg * l1_reg_loss

        # Calculate gradients
        label_loss.backward()
        # Optimizer
        optimizer.step()

        # Metrics calculation- For epoch Accuracy(total correct pred/total items) and loss 
        pred = label_pred.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        epoch_loss += label_loss.item()
        pbar.set_description(desc=f'Training Set: Loss={epoch_loss/len(train_loader)}, Batch_id={batch_id}, Train Accuracy={100*correct/processed:0.2f}')
    
    epoch_accuracy = (100*correct/processed)
    epoch_loss /= len(train_loader)
    # scheduler.step(epoch_loss/len(train_loader))
    
    scheduler.step(epoch_loss)

    return epoch_accuracy, epoch_loss

def train_model(data_name, model_name, resume, device, norm, epochs, batch_size, learning_rate):
    global start_epoch, best_acc
    # model = None
    train_accuracy = []
    test_accuracy = []
    train_loss = []
    test_loss = []

    if data_name == "CIFAR10":
        from models.resnet_32x32 import ResNet18
        train_loader, test_loader, classes = dataloaders("CIFAR10", train_batch_size=batch_size,
                                                        val_batch_size=batch_size,)

        if norm == "batch":
            model = ResNet18(use_batchnorm=True).to(device)
        elif norm == "layer":
            model = ResNet18(use_layernorm=True).to(device)
        elif norm == "group":
            model = ResNet18(use_groupnorm=True).to(device) 
        else:
            print("Please enter a valid Normalization Type")


    elif data_name == "CUSTOM":
        from models.resnet_224x224 import ResNet18
        train_loader, test_loader, classes, class_to_idx = dataloaders("CUSTOM", train_batch_size=batch_size,
                                                        val_batch_size=batch_size,)
        print(class_to_idx)
        model = ResNet18(pretrained = True, num_classes = len(classes)).to(device)
    image_shape = np.array(next(iter(train_loader))[0].shape[1:])                        
    print("\n\n****************************************************************************\n")
    print("*****Model Details*****\n")
    print(f"Input Image Size: {image_shape}\n")
    print(f"Model used: {ResNet18.__name__}\n")
    print("*****Training Parameters*****\n")
    print(f"Normalization Type: {norm} normalization\n")
    print(f"No Of Epochs: {epochs}\n")
    print(f"Batch size: {batch_size}\n")
    print(f"Initial Learning Rate: {learning_rate}")
    print("\n****************************************************************************\n")


    print("\n\n****************************************************************************\n")
    print("*****Model Summary*****")
    summary(model, input_size=(image_shape[0], image_shape[1], image_shape[2]))
    print("\n****************************************************************************\n")


    if resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/'+ model_name +'.pth')
        model.load_state_dict(checkpoint['net'])
        print('==> Model loaded from checkpoint..')
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']


    # Optimization algorithm from torch.optim
    optimizer = get_optimizer("SGD", model.parameters(), lr=learning_rate, weight_decay = 5e-4)
    scheduler = get_lr_sheduler(optimizer,"ReduceLROnPlateau",mode='min',factor=0.05, patience=2)
    # Loss condition
    criterion = nn.CrossEntropyLoss()
    print("\n****************************************************************************\n")
    print("*****Training Starts*****\n")

    for epoch in range(start_epoch + 1, start_epoch + epochs + 1):
        print(f"Training Epoch: {epoch}")
        train_acc_delta, train_loss_delta = train(model, device, train_loader, optimizer, scheduler, criterion)
        test_acc_delta, test_loss_delta, best_acc = test(model_name, model, device, test_loader, criterion, epoch, best_acc)
        # print(f"Learning Rate: {scheduler._last_lr[0]},{optimizer.param_groups[0]['lr']}")
        train_accuracy.append(round(train_acc_delta, 2))
        train_loss.append(round(train_loss_delta, 4))
        test_accuracy.append(round(test_acc_delta, 2))
        test_loss.append(round(test_loss_delta, 4))

    print("*****Training Stops*****\n")
    print("\n****************************************************************************\n")

    print("\n****************************************************************************\n")
    print("*****Loss and Accuracy Details*****\n")
    plot_single("model_1", train_loss, train_accuracy, test_loss, test_accuracy)
    print("\n****************************************************************************\n")

    return model