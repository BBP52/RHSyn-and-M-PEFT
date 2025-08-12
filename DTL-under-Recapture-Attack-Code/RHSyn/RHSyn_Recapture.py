import tqdm
import PIL.Image
from tqdm import tqdm
from natsort import natsorted
import os
import cv2
import numpy as np
import random
from PIL import Image
import multiprocessing
from tqdm import tqdm
from PIL import ImageCms


def rgb_to_cmyk_formula(rgb_array):
    """Converts RGB values to CMYK values."""
    r, g, b = rgb_array[:, :, 0], rgb_array[:, :, 1], rgb_array[:, :, 2]
    # b, g, r = rgb_array[:, :, 0], rgb_array[:, :, 1], rgb_array[:, :, 2]
    r = r / 255.0
    g = g / 255.0
    b = b / 255.0
    k = 1 - np.max(rgb_array / 255.0, axis=2)
    denominator = np.where(k == 1, 1, 1 - k)
    c = np.where(k == 1, 0, (1 - r - k) / denominator)
    m = np.where(k == 1, 0, (1 - g - k) / denominator)
    y = np.where(k == 1, 0, (1 - b - k) / denominator)
    cmyk_array = np.stack([c, m, y, k], axis=2)
    cmyk_array = cmyk_array*255
    cmyk_array = cmyk_array.astype(np.int)
    return cmyk_array


def rgb_to_cmyk_icc(rgb_array):
    
    ## choose icc file
    icc_file = './ICCFile/xx.icc'
    rgb_array = Image.fromarray(rgb_array)
    source_profile = ImageCms.createProfile("sRGB")
    target_profile = ImageCms.ImageCmsProfile(icc_file)
    transform = ImageCms.buildTransform(source_profile, target_profile, "RGB", "CMYK")
    converted_image = np.array(ImageCms.applyTransform(rgb_array, transform))
    converted_image = converted_image.astype(np.int64)
    return converted_image

## Error Diffusion
def errorDiffusionDithering(img):
    height, width = img.shape

    out = np.zeros((height, width), dtype = np.uint8)
    im = np.zeros((height + 2, width + 2), dtype = np.float32)
    im[1 : height + 1, 1 : width + 1] = img.astype(np.float32)

    fc = np.array([[0, 0, 7.0],
        [3.0, 5.0, 1.0]], dtype = np.float32)/16.0

    for y in range(1, height + 1):
        for x in range(1, width + 1):
            xc = im[y, x] > 128
            t = int(xc) * 255
            out[y - 1, x - 1] = t
            diff = t - im[y, x]
            fc1 = fc * diff
            im[y : y + 2, x - 1 : x + 2] -= fc1
    return out

def imggenerateFuc(image, imgCropList, imgCropListDict, gray_bins, imgCMYKDir):

    image_array = np.array(image)

    if 'Ink' in imgCMYKDir:
        ## For inkjet printers, error diffusion can be added to simulate the ink diffusion process.
        channel1 = image_array[:, :, 0]
        channel2 = image_array[:, :, 1]
        channel3 = image_array[:, :, 2]  
        out1 = errorDiffusionDithering(channel1)
        out2 = errorDiffusionDithering(channel2)
        out3 = errorDiffusionDithering(channel3)

        image_array[:, :, 0] = out1
        image_array[:, :, 1] = out2
        image_array[:, :, 2] = out3
        
        image_array = cv2.GaussianBlur(image_array, (7, 7), 0)

    ## ICC Color Conversion
    image_array_CMYK = rgb_to_cmyk_icc(image_array)

    image_result = image_array.copy()

    ## Convert to cmyk intensity range
    image_array_CMYK_new = image_array_CMYK / 255.0 * 100.0
    
    ## Channel-wise Quantization
    image_array_CMYK_new = np.digitize(image_array_CMYK_new, gray_bins, right = True)
    image_array_CMYK_new[image_array_CMYK_new == -1] += 1

    reshaped_arr = image_array_CMYK.reshape(-1, image_array_CMYK.shape[-1])
    reshaped_arr_new = reshaped_arr/ 255.0 * 100.0
    reshaped_arr_new = np.digitize(reshaped_arr_new, gray_bins, right = True)
    reshaped_arr_new[reshaped_arr_new == -1] += 1
    unique_image_array_CMYK = np.unique(reshaped_arr_new, axis = 0)

    CMYKPic = {}

    ## Mapping Process
    for CMYKPoint in unique_image_array_CMYK:
        C, M, Y, K = CMYKPoint
        RealC = C
        RealM = M
        RealY = Y
        RealK = K 

        channel_filePathC = "_".join(imgCropList[0][0].split("_")[:-1]) + "_" + str(RealC) + '.' + imgCropList[0][0].split('.')[-1]
        channel_filePathM = "_".join(imgCropList[1][0].split("_")[:-1]) + "_" + str(RealM) + '.' + imgCropList[1][0].split('.')[-1]
        channel_filePathY = "_".join(imgCropList[2][0].split("_")[:-1]) + "_" + str(RealY) + '.' + imgCropList[2][0].split('.')[-1]
        channel_filePathK = "_".join(imgCropList[3][0].split("_")[:-1]) + "_" + str(RealK) + '.' + imgCropList[3][0].split('.')[-1]

        channel_intensity_list = []
        channel_intensity_list.append(imgCropListDict[channel_filePathC])
        channel_intensity_list.append(imgCropListDict[channel_filePathM])
        channel_intensity_list.append(imgCropListDict[channel_filePathY])
        channel_intensity_list.append(imgCropListDict[channel_filePathK])

        CMYKPic[(C, M, Y, K)] = channel_intensity_list


    ## step_n can reduce the number of traversals for blocks of white pixels
    step_n = 16
    image_result_C = image_array.copy()
    image_result_M = image_array.copy()
    image_result_Y = image_array.copy()
    image_result_K = image_array.copy()
    for y in np.arange(0, image_array.shape[0], step_n):
        for x in np.arange(0, image_array.shape[1], step_n):
            CMYKArrayPixel = image_array_CMYK_new[y : y + step_n, x : x + step_n]

            CMYKArrayPixel1 = CMYKArrayPixel.reshape(-1, 4)
            unique_values = np.unique(CMYKArrayPixel1, axis = 0)
            if unique_values.shape[0] == 1:
                AddResult = CMYKPic[(CMYKArrayPixel[0, 0, 0], CMYKArrayPixel[0, 0, 1], CMYKArrayPixel[0, 0, 2], CMYKArrayPixel[0, 0, 3])]
                image_result_C[y : y + step_n, x : x + step_n] = AddResult[0][y : y + step_n, x : x + step_n]
                image_result_M[y: y + step_n, x: x + step_n] = AddResult[1][y: y + step_n, x: x + step_n]
                image_result_Y[y: y + step_n, x: x + step_n] = AddResult[2][y: y + step_n, x: x + step_n]
                image_result_K[y: y + step_n, x: x + step_n] = AddResult[3][y: y + step_n, x: x + step_n]
            else:
                for y1 in np.arange(step_n):
                    for x1 in np.arange(step_n):
                        AddResult = CMYKPic[(CMYKArrayPixel[y1, x1, 0], CMYKArrayPixel[y1, x1, 1], CMYKArrayPixel[y1, x1, 2], CMYKArrayPixel[y1, x1, 3])]
                        image_result_C[y + y1, x + x1] = AddResult[0][y + y1, x + x1]
                        image_result_M[y + y1, x + x1] = AddResult[1][y + y1, x + x1]
                        image_result_Y[y + y1, x + x1] = AddResult[2][y + y1, x + x1]
                        image_result_K[y + y1, x + x1] = AddResult[3][y + y1, x + x1]

    ## Element-wise Multiplication
    result = image_result_C/255.0 * image_result_M/255.0 * image_result_Y/255.0 * image_result_K/255.0

    image_result = np.uint8(result*255.0)

    ## Post-process
    ## Gamma Correction
    gamma = 1.3 # 伽马值
    normalized_image = image_result / 255.0
    corrected_image = np.power(normalized_image, gamma)
    image_result = np.clip((corrected_image * 255), 0, 255).astype(np.uint8)
 
    ## Blurring
    image_result = cv2.GaussianBlur(image_result, (3, 3), 0)

    ## More post-processing operations can be added...

    image = PIL.Image.fromarray(image_result)             

    return image



if __name__ == '__main__':

    ## Images to be recaptured
    imgDir = './image/'
    files = os.listdir(imgDir)
    random.shuffle(files)

    ## Reference Halftone Patterns
    imgCMYKRootDir = './ReferenceHalftonePatterns/'

    ## save root
    dstImgDir = './simulate/'

    if not os.path.exists(dstImgDir):
        os.makedirs(dstImgDir)

    imgCMYKDirList = [os.path.join(imgCMYKRootDir, i) for i in os.listdir(imgCMYKRootDir)]

    ## Different Print-and-Scan Devices
    for imgCMYKDir in imgCMYKDirList:
        device_name = os.path.basename(imgCMYKDir)
        print(f"\n{'=' * 50}\n正在处理设备: {device_name}\n{'=' * 50}")
        imgCropList = []
        imgCropListDict = {}

        imgCCropList = natsorted([i for i in os.listdir(imgCMYKDir) if "c" in i])
        for i in imgCCropList:
            imgCropListDict[i] = cv2.cvtColor(cv2.imread(os.path.join(imgCMYKDir, i)), cv2.COLOR_BGR2RGB)
        imgMCropList = natsorted([i for i in os.listdir(imgCMYKDir) if "m" in i])
        for i in imgMCropList:
            imgCropListDict[i] = cv2.cvtColor(cv2.imread(os.path.join(imgCMYKDir, i)), cv2.COLOR_BGR2RGB)

        imgYCropList = natsorted([i for i in os.listdir(imgCMYKDir) if "y" in i])
        for i in imgYCropList:
            imgCropListDict[i] = cv2.cvtColor(cv2.imread(os.path.join(imgCMYKDir, i)), cv2.COLOR_BGR2RGB)

        imgKCropList = natsorted([i for i in os.listdir(imgCMYKDir) if "k" in i])
        for i in imgKCropList:
            imgCropListDict[i] = cv2.cvtColor(cv2.imread(os.path.join(imgCMYKDir, i)), cv2.COLOR_BGR2RGB)

        ## CMYK Reference HalfTone Patterns
        imgCropList = [imgCCropList, imgMCropList, imgYCropList, imgKCropList]

        ## 40 intervals
        numList = [int(i.split(".")[0].split("_")[-1]) for i in imgCCropList]
        maxNumDuan = np.max(np.array(numList))
        pool = multiprocessing.Pool(processes=1)
        gray_bins = np.linspace(0, 100, num=(maxNumDuan + 1))

        ## Simulate Process
        for image_filename in tqdm(files, desc=f"Simulating for {device_name}"):
            image_path = os.path.join(imgDir, image_filename)
            original_image = Image.open(image_path)

            processed_block = imggenerateFuc(original_image, imgCropList, imgCropListDict, gray_bins, imgCMYKDir)

            new_filename = f"{device_name}_{image_filename}"

            new_filename_png = os.path.splitext(new_filename)[0] + '.png'

            output_path = os.path.join(dstImgDir, new_filename_png)

            processed_block.save(output_path)
            # ================================================================= #

        print(f"设备 '{device_name}' 处理完成。")

    














