import numpy as np
from PIL import Image
import cv2
import os
from pathlib import Path


# forked from https://gist.github.com/jinified/0287a81ddad4981e3a3a
def shades_of_gray(img, minkowski_norm=6.0):

    b,g,r = cv2.split(img)
    gray = np.mean([np.mean(b),np.mean(g),np.mean(r)])
    gray = np.power(gray, 1/minkowski_norm)
    r = gray/np.mean(r)*r
    r = np.uint8(cv2.normalize(r, 0, 255, cv2.NORM_MINMAX)*255)
    g = gray/np.mean(g)*g
    g = np.uint8(cv2.normalize(g, 0, 255, cv2.NORM_MINMAX)*255)
    b = gray/np.mean(b)*b
    b = np.uint8(cv2.normalize(b, 0, 255, cv2.NORM_MINMAX)*255)
    return cv2.merge((b,g,r))


# forked from https://github.com/sunnyshah2894/DigitalHairRemoval/blob/master/DigitalHairRemoval.py
def hair_removal(src):

    # Convert the original image to grayscale
    grayScale = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)

    # Kernel for the morphological filtering
    kernel = cv2.getStructuringElement(1, (17, 17))

    # Perform the blackHat filtering on the grayscale image to find the
    # hair countours
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)

    # intensify the hair countours in preparation for the inpainting
    # algorithm
    ret, thresh2 = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    # inpaint the original image depending on the mask
    dst = cv2.inpaint(src, thresh2, 1, cv2.INPAINT_TELEA)

    return dst


def mean_std_normalization(img):

    mean, std = cv2.meanStdDev(img)
    epsilon = 1e-20

    for i in range(mean.shape[0]):

        img[:, :, i] = (img[:, :, i] - mean[i]) / (std[i] + epsilon)

    return img

def preprocess(img_path):

    img = np.array(Image.open(os.path.join(img_path)))
    hair_removed = hair_removal(img)
    constancy = shades_of_gray(hair_removed)
    # constancy = constancy / 255

    return constancy


def single_thread_preprocess(orig_file):

    # preproc_path = '/home/ferles/Desktop/DL_Dermatology/DermatologyData/7-point/7_Point_Preprocessed'
    preproc_path = '/home/ferles/Desktop/SweDataset/Preproc'

    image_name = orig_file.split('/')[-1]
    processed_array = preprocess(orig_file)
    Image.fromarray(processed_array).save(os.path.join(preproc_path, image_name))


if __name__=='__main__':

    # TODO: Uncomment to preprocess training data from ISIC 2019
    # orig_path = '/home/ferles/Desktop/DL_Dermatology/DermatologyData/ISIC2019Data/ISIC_2019_Training_Input'
    # orig_files = sorted(glob.glob(os.path.join(orig_path, '*.jpg')))

    # TODO: Uncomment to preprocess test data from ISIC 2019
    # orig_path = '/home/ferles/Desktop/DL_Dermatology/DermatologyData/ISIC2019Data/ISIC_2019_Test_Input/'
    # orig_files = sorted(glob.glob(os.path.join(orig_path, '*.jpg')))

    # TODO: Uncomment to preprocess test data from ISIC 2019
    # orig_path = '/home/ferles/Desktop/DL_Dermatology/DermatologyData/Dermofit/data'
    # orig_files = [str(x) for x in list(Path(orig_path).glob('**/*.png'))]
    # orig_files = [x for x in orig_files if 'mask' not in x]

    # TODO: Uncomment to preprocess test data from ISIC 2019
    orig_path = '/home/ferles/Desktop/DL_Dermatology/DermatologyData/7-point/images/'
    orig_files = [str(x) for x in list(Path(orig_path).glob('**/*.jpg'))]
    orig_files = [x for x in orig_files if 'mask' not in x]

    orig_path = '/home/ferles/Desktop/SweDataset'
    orig_files = [str(x) for x in list(Path(orig_path).glob('**/*.jpg'))]
    import multiprocessing.dummy as mp
    # Set the number of threads
    p = mp.Pool(8)
    p.map(single_thread_preprocess, orig_files)
    p.close()
    p.join()

