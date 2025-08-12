import pickle
import cv2
import torchvision
from torch import optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from pathlib import Path
from contrastive.swin.swin_mmseg import SwinUper_mmseg
from contrastive.denseFCN.denseFCN import DenseFCN
from contrastive.beit.beit_mmseg import BeitUper_mmseg
from contrastive.tifdm.tifdm_base import Tifdm
from contrastive.convnext.Convnext_mmseg_Adapter import ConvnextUper_mmseg
from contrastive.poolformer.Poolformer_mmseg import Poolformer_mmseg
from contrastive.SegFormer.SegFormer_mmseg import SegFormer_mmseg
from contrastive.segNext.SegNext_mmseg import SegNext_mmseg
from losses import DiceLoss, SoftCrossEntropyLoss
import torch.nn as nn
import torch
from torch.cuda.amp import autocast as autocast, GradScaler
import random
import tempfile
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


## Train/Fine-tuning DataLoader
class TamperDataset(Dataset):
    def __init__(self, i0, data_path, steps = 8192, pilt = False, casia = False, ranger = 1, max_nums = None, max_readers = 64):
        
        self.i0 = i0
        self.steps = steps
        self.image_data = data_path
        self.img_file = read_content(self.image_data)
        random.shuffle(self.img_file)
        print('Training data size : {}'.format(len(self.img_file)))
        self.nSamples = len(self.img_file)

        np.random.seed(i0)

        self.hflip = torchvision.transforms.RandomHorizontalFlip(p=1.0)
        self.vflip = torchvision.transforms.RandomVerticalFlip(p=1.0)
        self.totsr = ToTensorV2()
        self.toctsr =torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize(mean=(0.485, 0.455, 0.406), std=(0.229, 0.224, 0.225))])

    def __len__(self):
        return self.nSamples

    def __getitem__(self, idx):
        
        img_path = self.img_file[idx]
        mask_path = img_path.replace('image', 'label')
        im = Image.open(img_path).convert('RGB')
        mask = cv2.imread(mask_path, 0)
        mask = (mask > 128).astype(np.uint8)
        H,W = mask.shape
        
        # DocTamper-Train used https://github.com/qcf-568/DocTamper
        if ((idx+self.i0)<600000):
            q = random.randint(100-np.clip((idx+self.i0)*random.uniform(0,1)//self.steps,0,25),100)
            q2 = random.randint(100-np.clip((idx+self.i0)*random.uniform(0,1)//self.steps,0,25),100)
            q3 = random.randint(100-np.clip((idx+self.i0)*random.uniform(0,1)//self.steps,0,25),100)
        else:
            q = random.randint(75,100)
            q2 = random.randint(75,100)
            q3 = random.randint(75,100)
        if random.uniform(0,1) < 0.5:
            im = im.rotate(90)
            mask = np.rot90(mask,1)
        mask = self.totsr(image = mask.copy())['image']
        if random.uniform(0,1) < 0.5:
            im = self.hflip(im)
            mask = self.hflip(mask)
        if random.uniform(0,1) < 0.5:
            im = self.vflip(im)
            mask = self.vflip(mask)
        with tempfile.NamedTemporaryFile(delete = True, prefix = str(idx)) as tmp:
            # DocTamper-Train used
            if 'DocTamper' in img_path:
                choicei = random.randint(0,2)
                if choicei > 1:
                    im.save(tmp, "JPEG", quality = q3)
                    im = Image.open(tmp)
                if choicei > 0:
                    im.save(tmp, "JPEG", quality = q2)
                    im = Image.open(tmp)
                im.save(tmp,"JPEG", quality=q)
                im = Image.open(tmp)
            im = self.toctsr(im)
            
        return {
            'image': im,
            'label': mask.long(),
            'filename' : img_path.split('/')[-1]
        }


## Val DataLoader
class TamperDatasetVal(Dataset):
    def __init__(self, data_path, minq = 95, qtb = 90, max_readers = 64):
        
        ## Testing Set 
        self.image_root = data_path
        self.img_file = read_content(self.image_root)    
        print('Testing data size : {}'.format(len(self.img_file)))
        self.nSamples = len(self.img_file)
        
        # DocTamper-Test used
        with open('./qt_table.pk','rb') as fpk:
            pks = pickle.load(fpk)
        self.pks = {}
        for k,v in pks.items():
            self.pks[k] = torch.LongTensor(v)
        with open('./pks/DocTamperV1-FCD'+'_%d.pk'%minq,'rb') as f:
            self.record = pickle.load(f)

        self.totsr = ToTensorV2()
        self.toctsr =torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=(0.485, 0.455, 0.406), std=(0.229, 0.224, 0.225))])

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        
        img_path = self.img_file[index]
        im = Image.open(img_path).convert('RGB')
        mask_path = img_path.replace('image', 'label')
        mask = cv2.imread(mask_path, 0)
        mask = (mask > 128).astype(np.uint8)
        H,W = mask.shape
        mask = self.totsr(image=mask.copy())['image']
        
        if 'DocTamper' in img_path:
            record = self.record[index]
            choicei = len(record)-1
            q = int(record[-1])
            use_qtb = self.pks[q]
            if choicei > 1:
                q2 = int(record[-3])
                use_qtb2 = self.pks[q2]
            if choicei > 0:
                q1 = int(record[-2])
                use_qtb1 = self.pks[q1]
                
        with tempfile.NamedTemporaryFile(delete=True) as tmp:
            if 'DocTamper' in img_path:
                if choicei > 1:
                    im.save(tmp,"JPEG",quality = q2)
                    im = Image.open(tmp)
                if choicei > 0:
                    im.save(tmp, "JPEG", quality = q1)
                    im = Image.open(tmp)
                im.save(tmp, "JPEG", quality = q)
                im = Image.open(tmp)
                
            im = self.toctsr(im)

        return {
            'image': im,
            'label': mask.long(),
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
        self.model_names = ['0_DTD', '1_BEiTUper','2_SwinUper','3_DenseFCN', '4_Tifdm', '5_ConvNeXt', '6_PoolFormer', '7_SegFormer'] 
        self.model_name = self.model_names[2]
        self.model_name = self.model_name.split('_')[-1]
        print('Train Model：'+ self.model_name)
        self.lr = 3e-5
        self.bottom_lr = 1e-5
        
        ## Save path
        self.checkpoint = Path('./result/')
        
        ## Train Path
        self.data_path = "./DocTamperV1-TrainingSet_image/"
        self.start_epoch = 0

        ## DocTamper-Train 10, Fine-Tuning 20
        self.epochs = 10
        
        ## Val Path
        self.data_path_val = "./DocTamperV1-FCD_image/"
        self.batch_size = 12
        self.numw = 8

                                           
print('Model reading phase')
'''
Model reading phase
'''

if __name__ == '__main__': 
    opt = Options()
                
    if opt.model_name == 'DTD':
        model = seg_dtd('', 2)
    elif opt.model_name == 'BEiTUper':
        model = BeitUper_mmseg()
    elif opt.model_name == 'SwinUper':
        model = SwinUper_mmseg()
    elif opt.model_name == 'DenseFCN':
        model = DenseFCN(2)
    elif opt.model_name == 'Tifdm':
        model = Tifdm()
    elif opt.model_name == 'ConvNeXt':
        model = ConvnextUper_mmseg()
    elif opt.model_name == 'PoolFormer':
        model = Poolformer_mmseg()
    elif opt.model_name == 'SegFormer':
        model = SegFormer_mmseg()

    ## Fine-tuning used
#     model_path = './result/xx.pth' 
#     ckpt = torch.load(model_path,map_location='cpu')
#     model.load_state_dict(ckpt['state_dict'], strict=False)
#     model.freeze_classifier_layers()

    device = opt.device
    model.to(device)

    for k,v in model.named_parameters():
        print('{}: {}'.format(k, v.requires_grad))
        
    train_data =  TamperDataset(0, opt.data_path)
       
    train_loader = iter(DataLoader(dataset=train_data, batch_size = opt.batch_size, num_workers = opt.numw))
    optimizer = optim.AdamW(model.parameters(), lr = opt.lr, weight_decay = 5e-4)
    
    train_data_size = train_data.__len__()
    iter_per_epoch = len(train_loader)
    totalstep = opt.epochs*iter_per_epoch
    warmstep = 200
    lr_min = opt.bottom_lr
    lr_min /= opt.lr
    lr_dict = {i:((((1+math.cos((i-warmstep)*math.pi/(totalstep-warmstep)))/2)+lr_min) if (i > warmstep) else (i/warmstep+lr_min)) for i in range(totalstep)}
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda epoch: lr_dict[epoch])
    DiceLoss_fn = DiceLoss(mode='multiclass')
    SoftCrossEntropy_fn = SoftCrossEntropyLoss(smooth_factor=0.1)
    scaler = GradScaler()
    model.train()  # Set the model to training mode
    for epoch in range(opt.start_epoch, opt.epochs):
        total_loss = 0
        avg_loss = 0
        tmp_i = epoch * train_data_size
        iter_i = epoch * iter_per_epoch
        if (epoch != 0):
            train_data = TamperDataset(tmp_i, opt.data_path)
            train_loader = iter(DataLoader(dataset = train_data, batch_size = opt.batch_size, num_workers = opt.numw))

        train_nums = [0]*len(train_loader)
        random.shuffle(train_nums)
        train_loader_size = len_train = len(train_nums)
        for batch_idx in tqdm(range(len(train_nums))):
            this_train_id = train_nums[batch_idx]
            if this_train_id==0:
                batch_samples = next(train_loader) 
            data, target, filename = batch_samples['image'], batch_samples['label'], batch_samples['filename']
            data, target = Variable(data.to(device)), Variable(target.to(device))
            with autocast():
                
                pred = model(data, filename)
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
        test_data = TamperDatasetVal(opt.data_path_val, minq = 75)
        test_loader1 = DataLoader(dataset=test_data, batch_size=12, num_workers=8, shuffle=False)
        model.eval()
        iou = IOUMetric(2)
        precisons = []
        recalls = []
        with torch.no_grad():
            for batch_idx, batch_samples in enumerate(tqdm(test_loader1)):
                data, target, filename = batch_samples['image'], batch_samples['label'], batch_samples['filename']
                data, target = Variable(data.to(device)), Variable(target.to(device))
                pred = model(data, filename)
                predt = pred.argmax(1)
                pred=pred.cpu().data.numpy()
                targt = target.squeeze(1)
                matched = (predt*targt).sum((1, 2))
                pred_sum = predt.sum((1, 2))
                target_sum = targt.sum((1, 2))
                precisons.append((matched/(pred_sum+1e-8)).mean().item())
                recalls.append((matched/target_sum).mean().item())
                pred = np.argmax(pred,axis=1)
                iou.add_batch(pred,target.cpu().data.numpy())
            acc, acc_cls, iu, mean_iu, fwavacc=iou.evaluate()
            precisons = np.array(precisons).mean()
            recalls = np.array(recalls).mean()
            precisons = np.array(precisons).mean()
            recalls = np.array(recalls).mean()
            print('[val] iou:{} pre:{} rec:{} f1:{}'.format(iu, precisons, recalls, (2*precisons*recalls/(precisons+recalls+1e-8))))

        state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        filename = os.path.join(opt.checkpoint, '{modelname}base-epoch{epoch}-iou{iou}.pth'.format(modelname = opt.model_name, epoch = epoch, iou = iu))
        torch.save(state, filename) 
