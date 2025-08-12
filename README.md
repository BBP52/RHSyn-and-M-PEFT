# DTL-under-Repcature-Attack

## Overview

This is the implementation of the method proposed in "Unmask Tampering: Efficient Document Tampering Localization under Recapturing Attacks with Real Distortion Knowledge" with pytorch(1.11.0, gpu version). The associated datasets are available upon request.

## Environment Request

python == 3.9.12  
torch == 1.11.0  
torchvision == 0.12.0  
opencv_python == 4.5.5.64  
Pillow == 9.0.1  
mmcv == 2.0.0  
mmcv-full == 1.6.2  
albumentations == 1.3.0  
timm == 0.4.12

## Files structure

* RHSyn
* M-PEFT

## Files Description

```RHSyn_Recapture.py```

Recapture step in RHSyn.

```BackBoneTrainBase.py```

pre-train a baseline model / Head-tuning / Full-tuning.

```BackBoneTrainMPEFT.py```

Masked Parameter-Efficient Fine-Tuning.

Ensure all files are placed correctly and named appropriately to enable successful model loading and data reading.

## Spoofing Dataset
You can download our spoofing dataset [here]( https://pan.baidu.com/s/10_U5ONvrvUpHmzDfSyvGPg). 🔑Extraction code: vrn7. 
