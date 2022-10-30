import albumentations as A
import glob
import random
# from torchvision import transforms

def get_transforms_SamllImage(train_mean_std=((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784)),
                    test_mean_std = ((0.49421428, 0.48513139, 0.45040909), (0.24665252, 0.24289226, 0.26159238)), image_res=32):
    train_transforms = A.Compose(
        [
            A.Normalize(train_mean_std[0], train_mean_std[1]),
            A.CropAndPad(px = int(image_res/8), keep_size=False),
            A.RandomCrop(width=image_res, height=image_res),
            A.CoarseDropout(1, int(image_res/2), int(image_res/2), 1, int(image_res/2), int(image_res/2),fill_value=0.473363),
            A.Rotate(5)
        ])

    val_transforms = A.Compose(
        [
            A.Normalize(mean=test_mean_std[0], std=test_mean_std[1]),
        ])
    return train_transforms, val_transforms

def get_transforms_CUSTOM(train_mean_std, test_mean_std, image_res):
    train_transforms = A.Compose(
        [
            A.CropAndPad(px = 16, keep_size=False),
            A.RandomCrop(width=image_res, height=image_res),
            A.CoarseDropout(2, 64, 64, 2, 64, 64,fill_value=0.4621),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(train_mean_std[0], train_mean_std[1]),
            A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5)
            
        ])

    val_transforms = A.Compose(
        [
            A.Normalize(mean=test_mean_std[0], std=test_mean_std[1]),
        ])
    return train_transforms, val_transforms
