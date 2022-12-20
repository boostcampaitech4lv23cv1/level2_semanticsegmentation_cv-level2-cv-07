import torch
import torch.nn as nn
import timm
import torchvision.models as models

class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        model = models.segmentation.fcn_resnet50(pretrained=True)
        model.classifier[4] = nn.Conv2d(512, 11, kernel_size=1)

    def forward(self, x):
        return x
    
if __name__ == "__main__":
    model = CustomModel()
    x = torch.randn([2, 3, 512, 512])
    print(f"input shape : {x.shape}")
    out = model(x)
    print(f"output shape : {out.size()}")