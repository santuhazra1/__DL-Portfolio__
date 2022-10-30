from torchvision import datasets
import torch
import numpy as np
import glob, random
import cv2
import matplotlib.pyplot as plt

from .transform import get_transforms_SamllImage, get_transforms_CUSTOM


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, train=True, train_data_path = 'data/train' , test_data_path = 'data/test', extension = 'png', transform=False, viz=False, calculate_mean_std=False):
        
        self.transform = transform
        self.viz = viz
        self.train=train
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.extension = extension
        if self.train:
            image_paths, class_to_idx, classes = self.get_custom_image_paths(self.train_data_path)
        else:
            image_paths, class_to_idx, classes = self.get_custom_image_paths(self.test_data_path)
        self.class_to_idx = class_to_idx
        self.classes = classes
        self.image_paths = image_paths
        if calculate_mean_std:
            self.mean, self.std = self.calculate_mean_std()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        image, label = self.read_image(self.image_paths[idx])
        if self.transform is not None:
            image = self.transform(image=np.array(image))["image"]
            if self.viz:
                return image, label
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)
            image = torch.tensor(image, dtype=torch.float)
        return image, label

    def get_custom_image_paths(self, data_paths):

        image_paths = [] #to store image paths in list
        classes = [] #to store class values

        for data_path in glob.glob(data_paths + '/*'):
            classes.append(data_path.split('/')[-1]) 
            image_paths.append(glob.glob(data_path + '/*.' + self.extension))
            
        image_paths = [image for images in list((image_paths)) for image in images]

        if self.train:
            random.shuffle(image_paths)

        idx_to_class = {i:j for i, j in enumerate(classes)}
        class_to_idx = {value:key for key,value in idx_to_class.items()}


        return image_paths, class_to_idx, classes

    def read_image(self, imagefile_path):
        image = cv2.imread(imagefile_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        
        label = imagefile_path.split('/')[-2]
        label = self.class_to_idx[label]

        return image, label
   
    def calculate_mean_std(self):
        psum    = torch.tensor([0.0, 0.0, 0.0])
        psum_sq = torch.tensor([0.0, 0.0, 0.0])

        for i in range(0,len(self.image_paths)):
            image, _ = self.read_image(self.image_paths[i])
            image = np.transpose(np.array(image), (2, 0, 1)).astype(np.float32)
            image = torch.tensor(image, dtype=torch.float)
            psum += image.sum(dim = [1, 2])
            psum_sq += (image ** 2).sum(axis = [1, 2])
        
        count = len(self.image_paths) * 224 * 224
        total_mean = psum / count
        total_var  = (psum_sq / count) - (total_mean ** 2)
        total_std  = torch.sqrt(total_var)

        return total_mean/255.0, total_std/255.0

class Cifar10Dataset(datasets.CIFAR10):
    
    def __init__(self, root = "./data", train = True, download = True, transform = None, viz = False):
        super().__init__(root = root, train = train, download = download, transform = transform)
        self.viz = viz
    
    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=np.array(image))
            image = transformed["image"]
            if self.viz:
                return image,label
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)
            image = torch.tensor(image, dtype=torch.float)
        
        return image,label


def dataloaders(data_name, train_batch_size = None, val_batch_size = None, seed=42):

    if data_name == "CIFAR10":
        train_transforms, test_transforms = get_transforms_SamllImage()
        train_ds = Cifar10Dataset('./data', train=True, download=True, transform=train_transforms)
        test_ds = Cifar10Dataset('./data', train=False, download=True, transform=test_transforms)

    elif data_name == "CUSTOM":
        train_transforms, test_transforms = get_transforms_CUSTOM(((0.5222, 0.4771, 0.3872), (0.2586, 0.2448, 0.2568)),
                                                            ((0.5220, 0.4850, 0.3979), (0.2602, 0.2461, 0.2621)), 224)
        train_ds = CustomDataset(train=True, transform=train_transforms)
        test_ds = CustomDataset(train=False, transform=test_transforms)
    
    cuda = torch.cuda.is_available()

    torch.manual_seed(seed)

    if cuda:
        torch.cuda.manual_seed(seed)
    
    train_batch_size = train_batch_size or (128 if cuda else 64)
    val_batch_size = val_batch_size or (128 if cuda else 64)

    train_dataloader_args = dict(shuffle=True, batch_size=train_batch_size, num_workers=4, pin_memory=True)
    val_dataloader_args = dict(shuffle=True, batch_size=val_batch_size, num_workers=4, pin_memory=True) 

    train_loader = torch.utils.data.DataLoader(train_ds, **train_dataloader_args)
    test_loader = torch.utils.data.DataLoader(test_ds, **val_dataloader_args)

    return train_loader, test_loader, train_ds.classes, train_ds.class_to_idx


def data_details(data_name, cols = 5, rows = 2, train_data = True, transform = False, vis=True): 

    if data_name == "CIFAR10":
        train_transforms, test_transforms = get_transforms_SamllImage()
        transform = train_transforms if train_data else test_transforms if transform else None
        data = Cifar10Dataset('./data', train=train_data, download=True, transform=transform, viz=vis )

    elif data_name == "CUSTOM":
        train_transforms, test_transforms = get_transforms_CUSTOM(((0.5222, 0.4771, 0.3872), (0.2586, 0.2448, 0.2568)),
                                                            ((0.5220, 0.4850, 0.3979), (0.2602, 0.2461, 0.2621)), 224)
        transform = train_transforms if train_data else test_transforms if transform else None
        data = CustomDataset(train=train_data, transform=transform, viz=vis)
    
    if vis:
        figure = plt.figure(figsize=(cols*1.5, rows*1.5))
        for i in range(1, cols * rows + 1):
            img, label = data[i]
            figure.add_subplot(rows, cols, i)
            plt.title(data.classes[label])
            plt.axis("off")
            plt.imshow(img, cmap="gray")

        plt.tight_layout()
        plt.show() 
    
    if (transform is None) and vis:
        if data_name != "CUSTOM":
            print(' - mean:', np.mean(data.data, axis=(0,1,2)) / 255.)
            print(' - std:', np.std(data.data, axis=(0,1,2)) / 255.)
        else:
            print(' - mean:', data.mean)
            print(' - std:', data.std)

    return data.classes

# data_details("CUSTOM", cols = 5, rows = 2, train_data = True, transform = True, vis=True)

# train_loader, _ = dataloaders("CUSTOM", train_batch_size = None, val_batch_size = None, seed=42)

# for image, label in train_loader:
#     print(image.shape)
#     print(label.shape)