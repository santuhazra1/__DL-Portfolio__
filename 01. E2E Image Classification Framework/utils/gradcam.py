import torch
import numpy as np
import torch.nn.functional as F

class Extractor:
    def __init__(self, model, layer_name):
        self.model = model.eval()
        self.gradients = []
        self.layer_name = layer_name
    
    def get_gradients(self):
        return self.gradients
    
    def save_gradient(self, grad):
        self.gradients.append(grad)
    
    def __call__(self, x):
        features = []
        self.gradients = []
        for name, module in self.model._modules.items():
            if name == "linear":
                x = F.avg_pool2d(x, 4)
                x = module(x.view(x.size(0), -1))
            else:
                x = module(x)
            if name == self.layer_name:
                x.register_hook(self.save_gradient)
                features.append(x)
        output = x
        return features, x

class GradCam:
    def __init__(self, extractor):
        self.extractor = extractor

    def __call__(self, x, index=None):
        """
        Args:
            - x: input, (1, C, H, W).
            - index: class index, (1,).
        Returns:
            - class activation map, (H', W').
            - extractor.model output, (1, C).
        """
        _, _, h, w = x.size()
        
        features, output = self.extractor(x)
        if index is None:
            index = output.argmax(dim=1).item()

        onehot = torch.zeros((1, output.size(1)), dtype=torch.float32, device=x.device)
        onehot[0,index] = 1.0
        onehot.requires_grad_(True)
        onehot = torch.sum(onehot * output)
        
        self.extractor.model.zero_grad()
        onehot.backward(retain_graph=True)
        
        grads = self.extractor.get_gradients()
        grads = grads[-1].detach().cpu().numpy()
        weights = np.mean(grads, axis=(2, 3), keepdims=True)
        features = features[-1].detach().cpu().numpy()
        cam = weights * features
        cam = np.sum(cam, axis=(0, 1))

        return cam, output   