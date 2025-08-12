import pickle
import cv2
import torchvision
from torch import optim
from torch.autograd import Variable
from pathlib import Path
from contrastive.swin.swin_mmseg_MPEFT import SwinUper_mmseg
from contrastive.beit.beit_mmseg_MPEFT import BeitUper_mmseg
from contrastive.poolformer.Poolformer_mmseg_MPEFT import Poolformer_mmseg
from contrastive.convnext.Convnext_mmseg_MPEFT import ConvnextUper_mmseg
from losses import DiceLoss, SoftCrossEntropyLoss
import torch
from torch.cuda.amp import autocast as autocast, GradScaler
import random
from PIL import Image
import math
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import os
from glob import glob
import numpy as np
from albumentations.pytorch import ToTensorV2

## Image loading
def read_content(path):
    if os.path.isdir(path):
        content = glob(os.path.join(path, '*.png')) + glob(os.path.join(path, '*.jpg'))
        
    elif os.path.isfile(path) and path.endswith('.txt'):
        with open(path, 'r') as file:
            content = file.read().splitlines()
    
        root_path = os.path.dirname(path)
        for i in range(len(content)):
            content[i] = os.path.join(root_path,content[i])
    else:
        content = None
        print('Unable to read the contents of the specified path...')
    return content

## Fine-tuning DataLoader
class TamperDataset(Dataset):
    def __init__(self, i0, data_path):
        
        self.i0 = i0
        self.image_data = data_path
        self.img_file = read_content(self.image_data)
        random.shuffle(self.img_file)
        print('Fine-tuning data size : {}'.format(len(self.img_file)))
        self.nSamples = len(self.img_file)

        np.random.seed(i0)

        self.hflip = torchvision.transforms.RandomHorizontalFlip(p = 1.0)
        self.vflip = torchvision.transforms.RandomVerticalFlip(p = 1.0)
        self.totsr = ToTensorV2()
        self.toctsr =torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=(0.485, 0.455, 0.406), std=(0.229, 0.224, 0.225))])

    def __len__(self):
        return self.nSamples

    def __getitem__(self, idx):
        
        img_path = self.img_file[idx]
        mask_path = img_path.replace('image', 'label')
        maskLabel_path = img_path.replace('image', 'maskLabel')
        im = Image.open(img_path).convert('RGB')
        mask = cv2.imread(mask_path, 0)
        mask = (mask > 128).astype(np.uint8)
        maskLabel = cv2.imread(maskLabel_path)
        
        if random.uniform(0,1) < 0.5:
            im = im.rotate(90)
            mask = np.rot90(mask, 1)
            maskLabel = np.rot90(maskLabel)
        mask = self.totsr(image = mask.copy())['image']
        maskLabel = self.totsr(image = maskLabel.copy())['image']
        if random.uniform(0,1) < 0.5:
            im = self.hflip(im)
            mask = self.hflip(mask)
            maskLabel = self.hflip(maskLabel)
        if random.uniform(0,1) < 0.5:
            im = self.vflip(im)
            mask = self.vflip(mask)
            maskLabel = self.vflip(maskLabel)
        
        im = self.toctsr(im)
        maskLabel = maskLabel / 255.0
        return {
            'image': im,
            'label': mask.long(),
            'mask' : maskLabel,
            'filename': img_path.split('/')[-1]
        }
    
## Val DataLoader
class TamperDatasetVal(Dataset):
    def __init__(self, data_path):
        
        self.image_root = data_path
        self.img_file = read_content(self.image_root)    
        print('Testing data size : {}'.format(len(self.img_file)))
        self.nSamples = len(self.img_file)
        self.totsr = ToTensorV2()
        self.toctsr = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=(0.485, 0.455, 0.406), std=(0.229, 0.224, 0.225))])

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        
        img_path = self.img_file[index]
        im = Image.open(img_path)
        mask_path = img_path.replace('image', 'label')
        maskLabel_path = img_path.replace('image', 'maskLabel')
        maskLabel = cv2.imread(maskLabel_path)
        mask = cv2.imread(mask_path, 0)
        mask = (mask > 128).astype(np.uint8)

        mask = self.totsr(image = mask.copy())['image']
        maskLabel = self.totsr(image = maskLabel.copy())['image']
        
        im = self.toctsr(im)
        maskLabel = maskLabel / 255.0
        
        return {
            'image': im,
            'label': mask.long(),
            'mask' : maskLabel,
            'filename': img_path.split('/')[-1]
        }
    
## Evaluation Metrics
class IOUMetric:
    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))
    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist
    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())
    def evaluate(self):
        acc = np.diag(self.hist).sum() / self.hist.sum()
        acc_cls = np.diag(self.hist) / self.hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        mean_iu = np.nanmean(iu)
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        return acc, acc_cls, iu, mean_iu, fwavacc

## Setup
class Options:
    def __init__(self):
        self.device = 'cuda:0'
        self.model_names = ['0_BEiTUper','1_SwinUper', '2_ConvNeXt', '3_PoolFormer'] 
        self.model_name = self.model_names[1]
        self.model_name = self.model_name.split('_')[-1]
        print('Fine-tuning Model：'+ self.model_name)
        self.lr = 3e-5
        self.bottom_lr = 1e-5
        
        ## Save path
        self.checkpoint = Path('./result/')
        
        ## Finetune Path
        self.data_path = "./SpoofSyn-Doc/image/"
        self.start_epoch = 0
        self.epochs = 20
        
        ## Val Path
        self.data_path_val = "./SpoofCert-HQ/image/"
        self.batch_size = 12
        self.numw = 8
        self.maskwidthBack = 64
        self.maskwidthFore = 64

        
print('Model reading phase')
'''
Model reading phase
'''

if __name__ == '__main__': 
    opt = Options()
                      
    if opt.model_name == 'BEiTUper':
        model = BeitUper_mmseg(opt.maskwidthBack, opt.maskwidthFore)
    elif opt.model_name == 'SwinUper':
        model = SwinUper_mmseg(opt.maskwidthBack, opt.maskwidthFore)
    elif opt.model_name == 'ConvNeXt':
        model = ConvnextUper_mmseg(opt.maskwidthBack, opt.maskwidthFore)
    elif opt.model_name == 'PoolFormer':
        model = Poolformer_mmseg(opt.maskwidthBack, opt.maskwidthFore)

    ## BaseLine Model
    model_path = './result/xx.pth' 
    ckpt = torch.load(model_path, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model.freeze_classifier_layers()

    device = opt.device
    model.to(device)

    for k,v in model.named_parameters():
        print('{}: {}'.format(k, v.requires_grad))
        
    train_data =  TamperDataset(0, opt.data_path)
    
    train_loader = iter(DataLoader(dataset=train_data, batch_size=opt.batch_size, num_workers=opt.numw))
    optimizer = optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=5e-4)
    
    train_data_size = train_data.__len__()
    iter_per_epoch = len(train_loader)
    totalstep = opt.epochs * iter_per_epoch
    warmstep = 200
    lr_min = opt.bottom_lr
    lr_min /= opt.lr
    lr_dict = {i:((((1+math.cos((i-warmstep)*math.pi/(totalstep-warmstep)))/2)+lr_min) if (i > warmstep) else (i/warmstep+lr_min)) for i in range(totalstep)}
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: lr_dict[epoch])
    DiceLoss_fn = DiceLoss(mode = 'multiclass')
    SoftCrossEntropy_fn = SoftCrossEntropyLoss(smooth_factor = 0.1)
    scaler = GradScaler()
    model.train()  # Set the model to training mode
    
    for epoch in range(opt.start_epoch, opt.epochs):
        total_loss = 0
        avg_loss = 0
        tmp_i = epoch*train_data_size
        iter_i = epoch*iter_per_epoch
        if (epoch!=0):
            train_data = TamperDataset(tmp_i, opt.data_path)
            train_loader = iter(DataLoader(dataset = train_data, batch_size = opt.batch_size, num_workers = opt.numw))

        train_nums = [0]*len(train_loader)
        random.shuffle(train_nums)
        train_loader_size = len_train = len(train_nums)
        for batch_idx in tqdm(range(len(train_nums))):
            this_train_id = train_nums[batch_idx]
            if this_train_id==0:
                batch_samples = next(train_loader)
            data, target, mask, filename = batch_samples['image'], batch_samples['label'], batch_samples['mask'], batch_samples['filename']
            data, target, mask = Variable(data.to(device)), Variable(target.to(device)), Variable(mask.to(device))
            with autocast():
                
                pred = model(data, mask, filename)
                loss = 1.*DiceLoss_fn(pred, target) + SoftCrossEntropy_fn(pred, target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            scheduler.step(iter_i+batch_idx) 
            total_loss += loss.item()
            if batch_idx % 500 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx} loss: {loss.item()}")
                
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} Average loss: {avg_loss}")
        
        ## Testing Phase
        test_data = TamperDatasetVal(opt.data_path_val)
        test_loader = DataLoader(dataset = test_data, batch_size = 12, num_workers = 8, shuffle = False)
        model.eval()
        iou = IOUMetric(2)
        precisons = []
        recalls = []
        
        with torch.no_grad():
            for batch_idx, batch_samples in enumerate(tqdm(test_loader)):
                
                data, target, mask, filename = batch_samples['image'], batch_samples['label'], batch_samples['mask'], batch_samples['filename']
                data, target, mask = Variable(data.to(device)), Variable(target.to(device)), Variable(mask.to(device))
                pred = model(data, mask, filename)   
                predt = pred.argmax(1)
                pred = pred.cpu().data.numpy()
                targt = target.squeeze(1)
                matched = (predt * targt).sum((1, 2))
                pred_sum = predt.sum((1, 2))
                target_sum = targt.sum((1, 2))
                precisons.append((matched / (pred_sum + 1e-8)).mean().item())
                recalls.append((matched / target_sum).mean().item())
                pred = np.argmax(pred, axis=1) 
                iou.add_batch(pred, target.cpu().data.numpy())

            acc, acc_cls, iu, mean_iu, fwavacc = iou.evaluate()
            precisons = np.array(precisons).mean()
            recalls = np.array(recalls).mean()
            precisons = np.array(precisons).mean()
            recalls = np.array(recalls).mean()
            print('[val] iou:{} pre:{} rec:{} f1:{}'.format(iu, precisons, recalls, (2 * precisons * recalls / (precisons + recalls + 1e-8))))

        state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        filename = os.path.join(opt.checkpoint, '{modelname}MPEFT-epoch{epoch}-iou{iou}.pth'.format(modelname = opt.model_name, epoch = epoch, iou = iu))
        torch.save(state, filename) 
