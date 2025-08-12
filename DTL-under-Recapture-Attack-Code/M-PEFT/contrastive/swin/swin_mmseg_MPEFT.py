import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

global_mask = None
global_filename = None

## M-Adapter
class ModifiedLinear(nn.Module):
    def __init__(self, proj, stage_idx, block_idx, mlp_ratio = 4, act_layer = nn.GELU, skip_connect = True):
        super().__init__()

        self.stage_idx = stage_idx
        self.block_idx = block_idx
        if stage_idx == 0:
            self.patWidth = 128
            self.featureSize = 128

        elif stage_idx == 1:
            self.patWidth = 64
            self.featureSize = 256

        elif stage_idx == 2:
            self.patWidth = 32
            self.featureSize = 512

        elif stage_idx == 3:
            self.patWidth = 16
            self.featureSize = 1024
        
        self.gamma = nn.Identity()
        
        self.skip_connect = True
        hidden_features = int(self.featureSize // mlp_ratio)
        self.act = act_layer()
        self.fc1 = nn.Linear(self.featureSize, hidden_features)
        self.fc2 = nn.Linear(hidden_features, self.featureSize)
        

    def forward(self, x):
        x = self.gamma(x)

        global global_mask 
        mask = global_mask
        
        global global_filename 
        filename = global_filename
        
        ## Match the Feature Size
        mask = F.interpolate(mask, size=(self.patWidth, self.patWidth), mode='bilinear', align_corners=False)
        mask[mask != 0] = 1
        tensor_single_channel = mask[:, 0, :, :]
        mask = tensor_single_channel.unsqueeze(1).repeat(1, self.featureSize, 1, 1)
        
        original_x = x
        
        x = x.view(x.size(0), self.patWidth, self.patWidth, -1)      
        x = x.permute(0, 3, 1, 2)
        x_fore = x.clone()
        x_fore[mask == 0] = 0
        x_fore = x_fore.permute(0, 2, 3, 1)  
        x_fore = x_fore.view(x_fore.size(0), -1, x_fore.size(-1))
        
        x = original_x
        xs = self.fc1(x_fore)
        xs = self.act(xs)
        xs = self.fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
               
        return x
    
class SwinUper_mmseg(nn.Module):
    def __init__(self, maskwidthBack = 512, maskwidthFore = 512):
        super(SwinUper_mmseg, self).__init__()

        model_save = './swin_upernet_mmseg_base.pth'
        self.model = torch.load(model_save, map_location='cpu')
        for name, module in self.model.named_modules():
            if name.split('.')[-1] == 'gamma2':
                stage_idx = int(name.split('.')[2])
                block_idx = int(name.split('.')[4])
                self.model.backbone.stages[stage_idx].blocks[block_idx].ffn.gamma2 = ModifiedLinear(module, stage_idx, block_idx)
    
        self.mask_amp_fore = nn.Parameter(data = torch.zeros(1, 1, maskwidthFore, maskwidthFore), requires_grad = True)
        self.mask_pha_fore = nn.Parameter(data = torch.zeros(1, 1, maskwidthFore, maskwidthFore), requires_grad = True)
        self.mask_amp_back = nn.Parameter(data = torch.zeros(1, 1, maskwidthBack, maskwidthBack), requires_grad = True)
        self.mask_pha_back = nn.Parameter(data = torch.zeros(1, 1, maskwidthBack, maskwidthBack), requires_grad = True)

    def get_centralized_spectrum(self, img):
        img_fft = torch.fft.fft2(img, s=None, dim=(-2, -1),norm='ortho')
        img_fft_shift = torch.fft.fftshift(img_fft)
        return img_fft_shift
    
    def get_ifft(self, mask_amp, mask_pha, img_fft):
        h = img_fft.shape[2]
        w = img_fft.shape[3]
        h_start_amp = h//2 - mask_amp.shape[2]//2
        w_start_amp = w//2 - mask_amp.shape[3]//2
        h_end_amp = h//2 + mask_amp.shape[2]//2
        w_end_amp = w//2 + mask_amp.shape[3]//2

        h_start_pha = h//2 - mask_pha.shape[2]//2
        w_start_pha = w//2 - mask_pha.shape[3]//2
        h_end_pha = h//2 + mask_pha.shape[2]//2
        w_end_pha = w//2 + mask_pha.shape[3]//2

        new_fft = img_fft

        real = torch.real(img_fft)
        imag = torch.imag(img_fft)

        real[:, :, h_start_amp:h_end_amp, w_start_amp:w_end_amp] = mask_amp + real[:, :, h_start_amp:h_end_amp, w_start_amp:w_end_amp]
        imag[:, :, h_start_pha:h_end_pha, w_start_pha:w_end_pha] = mask_pha + imag[:, :, h_start_pha:h_end_pha, w_start_pha:w_end_pha]
        new_fft = real + 1j*imag
        new_fft = torch.fft.ifftshift(new_fft) 
        new_img = torch.fft.ifft2(new_fft, s = None, dim = (-2, -1), norm = 'ortho').real
        return new_img

        
    def forward(self, x, mask, filename):
        
        global global_mask
        global global_filename
        
        global_mask = mask
        global_filename = filename
        
        mask_amp_fore = self.mask_amp_fore
        mask_pha_fore = self.mask_pha_fore
        mask_amp_back = self.mask_amp_back
        mask_pha_back = self.mask_pha_back
        
        ## M-FVP
        x_fore = x.clone()
        x_background = x.clone()
        x_fore[mask == 0] = 0
        x_background[mask == 1] = 0
        img_fft_fore = self.get_centralized_spectrum(x_fore)
        img_ifft_fore = self.get_ifft(mask_amp_fore, mask_pha_fore, img_fft_fore)
        img_fft_back = self.get_centralized_spectrum(x_background)
        img_ifft_back = self.get_ifft(mask_amp_back, mask_pha_back, img_fft_back)
        merged_tensor = torch.zeros_like(x_fore)
        merged_tensor[x_fore == 0] = 0
        merged_tensor[x_background == 0] = 1
        images = merged_tensor * img_ifft_fore + (1 - merged_tensor) * img_ifft_back

        output = self.model(images)
        output = F.interpolate(output, size=512, mode='bilinear', align_corners=False)
        return output
       
    def freeze_classifier_layers(self):
        for name, param in self.model.named_parameters():
            if 'decode_head' in name or 'fc1' in name or 'fc2' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

