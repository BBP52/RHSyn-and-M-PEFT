import torch
import torch.nn.functional as F
import torch.nn as nn

class SwinUper_mmseg(nn.Module):
    def __init__(self):
        super(SwinUper_mmseg, self).__init__()
        ## https://github.com/open-mmlab/mmsegmentation/tree/main/configs/swin
        ## swin-base-patch4-window12-in1k-384x384-pre_upernet_8xb2-160k_ade20k-512x512
        model_save = './swin_upernet_mmseg_base.pth'
        self.model = torch.load(model_save, map_location='cpu')
        
    def forward(self, x, filename):
        output = self.model(x)
        output = F.interpolate(output, size=512, mode='bilinear', align_corners=False)
        return output
    
    def freeze_classifier_layers(self):
        for name, param in self.model.named_parameters():
            if 'decode_head' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
