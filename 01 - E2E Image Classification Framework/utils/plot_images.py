import torch
import matplotlib.pyplot as plt
from .dataloader import dataloaders, data_details
from .gradcam import Extractor, GradCam
import numpy as np
import cv2

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    # heatmap = np.float32(heatmap) / 255
    # cam = (0.4* heatmap) + (0.6 * np.float32(img))
    heatmap = heatmap / np.max(heatmap)
    return np.uint8(255 * heatmap)

def plot_pred_images(data, title, classes, r=5,c=4):
    fig, axs = plt.subplots(r,c,figsize=(15,10))
    fig.tight_layout()
    # fig.suptitle(title)
    for i in range(r):
        for j in range(c):
            axs[i][j].axis('off')
            axs[i][j].set_title(f"Target: {classes[int(data[(i*c)+j]['target'])]}\nPred: {classes[int(data[(i*c)+j]['pred'])]}")
            axs[i][j].imshow(inverse_normalize(data[(i*c)+j]['data']).squeeze().cpu().permute(1,2,0))
    plt.show()

def plot_gradcam_images(model, data, title, classes, r=5,c=4):
    fig, axs = plt.subplots(r,c,figsize=(15,10))
    fig.tight_layout()
    # fig.suptitle(title)
    grad_cam = GradCam(Extractor(model, "layer3"))
    for i in range(r):
        for j in range(c):
            grad_cam_img,output = grad_cam(data[(i*c)+j]['data'])
            grad_cam_img = (grad_cam_img - np.min(grad_cam_img)) / (np.max(grad_cam_img) - np.min(grad_cam_img))
            grad_cam_img = cv2.resize(grad_cam_img, (32, 32))
            image = inverse_normalize(data[(i*c)+j]['data']).squeeze().cpu().permute(1,2,0)
            axs[i][j].axis('off')
            axs[i][j].set_title(f"Target: {classes[int(data[(i*c)+j]['target'])]}\nPred: {classes[int(data[(i*c)+j]['pred'])]}")
            axs[i][j].imshow(image)
            axs[i][j].imshow(grad_cam_img, alpha=0.3, cmap='jet')
    plt.show()

def inverse_normalize(tensor, mean=(0.49139968, 0.48215841, 0.44653091), std=(0.24703223, 0.24348513, 0.26158784)):
  # Not mul by 255 here
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def image_prediction(data_name, model, title, n=20,r=5,c=4, misclassified = True, gradcam=False):
    model.eval()
    _, test_loader, _ = dataloaders(data_name, val_batch_size=1)
    classes = data_details(data_name, cols=5, rows=2, train_data=True, transform=False, vis = False)
    wrong = []
    right = []
    i, j = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).item()
            if (not correct) and i <= n - 1:
                wrong.append({
                    "data": data,
                    "target": target.item(),
                    "pred": pred.item()
                })
                i+=1
            elif j <= n - 1:
                right.append({
                    "data": data,
                    "target": target.item(),
                    "pred": pred.item()       
                })
                j+=1

    if not(gradcam) and misclassified:
        plot_pred_images(wrong, title, classes, r, c)

    elif not(gradcam) and not(misclassified):
        plot_pred_images(right, title, classes, r, c)

    elif gradcam and misclassified:
        plot_gradcam_images(model, wrong, title, classes, r, c)

    else:
        plot_gradcam_images(model, right, title, classes, r, c)