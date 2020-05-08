import numpy as np
import mdlParams
num_images = 0
ncrops = 16
image_size = (224, 224)
test_image_size = (224, 224)
crop_size=224

mdlParams['cropPositions'] = np.zeros(num_images, ncrops, 2)
for u in range(num_images):
    height, width = image_size
    if width < crop_size:
        height = int(crop_size/float(width)) * height
        width = crop_size
    if height < crop_size:
        width = int(crop_size/float(height))*width
        height = crop_size
    ind = 0
    for i in range(np.sqrt(ncrops)):
        for j in range(np.sqrt(ncrops)):
            mdlParams['cropPositions'][u, ind, 0] = crop_size/2 + i*(width-crop_size)/(np.sqrt(ncrops)-1)
            mdlParams['cropPositions'][u, ind, 1] = crop_size/2 + j*(height-crop_size)/(np.sqrt(ncrops)-1)
            ind += 1

height = crop_size
width = crop_size
for u in range(num_images):
    height_test, width_test = test_image_size
    if width_test < crop_size:
        height_test = int(crop_size/float(width_test))*height_test
        width_test = crop_size
    if height_test < crop_size:
        width_test = int(crop_size/float(height_test))*width_test
        height_test = crop_size
    test_im = np.zeros([width_test, height_test])
    for i in range(mdlParams['multiCropEval']):
        im_crop = test_im[  np.int32(mdlParams['cropPositions'][u, i, 0]-height/2):
                            np.int32(mdlParams['cropPositions'][u, i, 0]-height/2)+height,
                            np.int32(mdlParams['cropPositions'][u, i, 1]-width/2):
                            np.int32(mdlParams['cropPositions'][u, i, 1]-width/2)+width]
        if im_crop.shape[0] != mdlParams['input_size'][0]:
            print("Wrong shape",im_crop.shape[0],mdlParams['im_paths'][u])
        if im_crop.shape[1] != mdlParams['input_size'][1]:
            print("Wrong shape",im_crop.shape[1],mdlParams['im_paths'][u])