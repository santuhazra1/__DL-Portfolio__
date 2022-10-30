import torch.optim as optim

def get_lr_sheduler(optimizer, lr_type = "", lr=0.001, mode='min',factor=0.05, patience=2):
    if lr_type == "ReduceLROnPlateau":
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.05, patience=2)
    elif lr_type == "CyclicLR":
        return optim.lr_scheduler.CyclicLR(optimizer, base_lr = 0.0001, max_lr=0.01, step_size_up=2000, step_size_down=None, mode='triangular')
