import torch.optim as optim

def get_optimizer(optim_type, params, lr=0.001, momentum=0.9, weight_decay=0):
    if optim_type == "SGD":
        return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optim_type == "ADAM":
        return optim.Adam(params, lr=lr)

