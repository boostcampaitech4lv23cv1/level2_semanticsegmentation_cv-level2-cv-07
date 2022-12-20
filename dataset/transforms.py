import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import json
class Transform:
    train = A.Compose([
        ToTensorV2()
        ])

    val = A.Compose([
        ToTensorV2()
        ])

    test = A.Compose([
        ToTensorV2()
        ])

if __name__ == "__main__":
    Transform().train
    Transform().val
    Transform().test