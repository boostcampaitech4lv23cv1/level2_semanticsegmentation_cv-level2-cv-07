import torch
import torch.nn as nn
import timm
import torchvision.models as models
import segmentation_models_pytorch as smp



class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Example for using SMP
        self.model = smp.Unet(
            encoder_name="timm-mobilenetv3_large_100",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=11,                      # model output channels (number of classes in your dataset)
        )

        
        # Example for using torchvision segmentation
        ## predict 시 ['out'] 키 호출 필요
        # self.model = models.segmentation.fcn_resnet50(pretrained=False)
        # self.model.classifier[4] = nn.Conv2d(512, 11, kernel_size=1)

    def forward(self, x):
        x = self.model(x)
        return x
    
if __name__ == "__main__":
    model = CustomModel()
    x = torch.randn([2, 3, 512, 512])
    print(f"input shape : {x.shape}")
    out = model(x)
    print(f"output shape : {out.size()}")