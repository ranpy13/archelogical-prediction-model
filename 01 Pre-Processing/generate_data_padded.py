"""
This file generates data any class [aguada, building, platform] by reading in the corresponding mask,
getting the position of the class pixels,  
pad it using grey_erosion, 
copy the corresponding pixel values into an array
take a random picture with no class element
and randomly place those into the empty image.
Finally, it saves both the generated image and the corresponding mask.


Within the function 
"""

#%% imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import random
from scipy.ndimage import grey_erosion

#%% read in meta data to know which files contain aguada images
# change root into your local directory
root = 'C:/Users/joa24jm/Documents/maya_mystics/' # change to your root dir
df = pd.read_csv(root + 'data/00_meta/maya_analysis.csv')

#%% helper functions
def get_file_numbers(df, cla=None):
    """
    Parameters
    ----------
    df : dataframe with columns 
        name  (filename),
        class (aguada, building, platform)
        pix_count (if > 0, class is in that picture)
    cla : class of interest, either aguada, building or platform
        default = None (returns then all numbers with no classes in it)
    
    Returns
    -------
    list with all filenumbers containing that class
    """
   
    # get all aguada images
    files = df[(df['class'] == cla) & (df.pixCount > 0)].name.values.tolist()
    
    if cla == None:
        # get all image numbers with no class at all
        files = df[df.pixCount == 0].name.values.tolist()
        # keep only number if there is no class at all in this picture
        files = [f for f in files if files.count(f) == 3]
        # remove duplicates
        files = list(set(files))
    
    # safe only file number 
    files = [f.split('_')[1] for f in files]
    
    return files

#%% read in sample of masks

def show_img(filenumber):
    """
    Parameters
    ----------
    filenumber : number of file to display

    Returns
    -------
    Pixel values of this image

    """
    n = filenumber
    

    
    paths = []
    pixs = []
    for c in ['aguada', 'building', 'platform']:
        paths.append(Image.open(root + 'data/01_raw/train_masks/' + f'tile_{n}_mask_{c}.tif'))
        pixs.append(np.array(paths[-1]))
    
    fig , axs = plt.subplots(2, 3, sharex = True, sharey = True)
    for i,c in enumerate(['aguada', 'building', 'platform']):
        axs[1,i].imshow(pixs[i], cmap='gray', vmin=0, vmax=255.)
        axs[1,i].set_title(c)
        

    img = Image.open(root + 'data/01_raw/lidar_train/' + f'tile_{n}_lidar.tif')
    axs[0,1].imshow(img)
    axs[0,1].set_title('Ground Truth')
        
    plt.suptitle(f'tile number {n}')
    plt.show()
    
def return_np_array(filenumber, mask=True, cla='aguada'):
    """
    filenumber as string
    mask as boolean, True or False
    class as string 'aguada', 'building', 'platform'
    
    Returns:
        np.array
    """
    
    n = filenumber
    c = cla # class aguada, building, platform
    
    if mask: # return one channel grayscale array
        img = Image.open(root + 'data/01_raw/train_masks/' + f'tile_{n}_mask_{c}.tif')
    
    else: # return rgb channel
        img = Image.open(root + 'data/01_raw/lidar_train/' + f'tile_{n}_lidar.tif')
    
    return(np.array(img))


#%% inspect values of pix array

def get_bounding_box(pix):
    """
    Parameters
    ----------
    pix : numpy array of mask raw values

    Returns
    -------
    bounding box for mask 

    """
    # loop over rows
    for r1 in range(pix.shape[0]):   
        # get top row where first mask pixel appears
        if 0 in np.unique(pix[r1, :]):
            top_row = r1
            break
    
    # loop over rows
    for r2 in range(pix.shape[0]):
        # get bottom row by starting from the last row and go upwards
        if 0 in np.unique(pix[pix.shape[0]-r2-1]):
            bottom_row = pix.shape[0]-r2
            break
    
    # loop over cols       
    for c1 in range(pix.shape[1]):
        # get left column where first mask pixel appears
        if 0 in np.unique(pix[:, c1]):
            left_col = c1
            break
    
    # loop over cols
    for c2 in range(pix.shape[1]):
        # get right column by starting from the last col and go upwards
        if 0 in np.unique(pix[:, pix.shape[1]-c2-1]):
            right_col = pix.shape[1]-c2
            break

    return (top_row, bottom_row, left_col, right_col)

def get_img_slice(bbox, pix, mask = False):
    """
    Parameters
    ----------
    bbox : bounding box as tuple with shape (top_row, bottom_row, left_col, right_col)
    pix : numpy array of shape (width, height, n_channels)
    mask: Boolean, Indicates wheter to take a slice from as mask or an image

    Returns
    -------
    cropped image of shape (widht_cropped, height_cropped, n_channels)

    """
    if mask:
        return pix[bbox[0]:bbox[1], bbox[2]:bbox[3]]
    else:
        return pix[bbox[0]:bbox[1], bbox[2]:bbox[3], :]
    
def pad_with(vector, pad_width, iaxis, kwargs):
    
    """
    Pad array with values of choice
    https://numpy.org/doc/stable/reference/generated/numpy.pad.html
    """

    pad_value = kwargs.get('padder', 10)

    vector[:pad_width[0]] = pad_value

    vector[-pad_width[1]:] = pad_value
    

def generate_training_data(image_slice, mask_slice, target, bbox, padding):
    """
    Parameters
    ----------
    cropped_slice : pixel values of shape (width, height, n_channels)
    target : pixel values of shape (width, height, n_channels)
    padding: Boolean: Indicates, whether masks can put at the picture margin 

    Returns
    -------
    Image with randomly placed class in it

    """
    # calculate h and w with h as vertical padding and w as horizontal padding
    if padding:
        h = bbox[1] - bbox[0]
        w = bbox[3] - bbox[2]
    else:
        h = 0
        w = 0
    
    # get dimension of target_image (should be 480, 480, 3)
    total_y = target.shape[0] # total height
    total_x = target.shape[1] # total width
    
    # calculate a valid random position for cropped_slice
    x_pos = random.choice(np.arange(0, total_y-h, 1))
    y_pos = random.choice(np.arange(0, total_x-w, 1))
    
    # check how to crop the image slice
    img_y = image_slice.shape[0] # vertical axis
    img_x = image_slice.shape[1] # horizontal axis
    
    # crop image slice on horizontal axis x
    if x_pos + img_x > total_x:
        image_slice = image_slice[:, :total_x - x_pos, :]
        mask_slice = mask_slice[:, :total_x - x_pos]
    
    # crop image slice on vertical axis y
    if y_pos + img_y > total_y:
        image_slice = image_slice[:total_y - y_pos, :, :]
        mask_slice = mask_slice[:total_y - y_pos, :]
           
    # 0. Pad mask with 0s
    mask_slice_for_image_crop = grey_erosion(mask_slice, size=(8,8))
    
    # 1. Expand dimension
    mask_slice_3 = np.array([mask_slice_for_image_crop, mask_slice_for_image_crop, mask_slice_for_image_crop])
    # 2. Reorder channels to (height, width, channels)
    mask_slice_3 = np.transpose(mask_slice_3, (1, 2, 0))
    # 3. Use np.place to replace those values in target for which 
    # the mask values == 0 with the image_slice
    
    # place cropped_slice into target
    np.place(target[y_pos:y_pos+image_slice.shape[0], 
                    x_pos:x_pos+image_slice.shape[1], :], 
             mask_slice_3 == 0,
             image_slice)
    
    new_image = target
    new_mask = generate_new_mask(x_pos, y_pos, mask_slice)
    
    
    return new_image, new_mask

def generate_new_mask(x_pos, y_pos, mask_slice):
    """
    Parameters
    ----------
    x_pos : same x_pos to insert mask like in image
    y_pos : same y_pos to insert mask like in image
    pix_mask : the mask containing the class to place into the empty mask

    Returns
    -------
    new_mask : new mask that matches the new generated image

    """
    
    # generate an empty mask
    empty_mask = np.full(shape = (480, 480), fill_value = 255)
        
    empty_mask[y_pos:y_pos+mask_slice.shape[0], x_pos:x_pos+mask_slice.shape[1]] = mask_slice
    
    new_mask = empty_mask
    
    return new_mask

#%% get file numbers with no classes
import os
# from which class shall images be generated?
cla = 'aguada'

# all numbers with original images containing that class
class_files = get_file_numbers(df, cla=cla)
# all numbers from empty images
no_classes = get_file_numbers(df, cla=None)


for i, file in enumerate(no_classes):

    print(i)    
    # select random from an empty image with no classes
    plain_image = file
    
    class_file = random.choice(class_files)
    
    # get numpy data from this random image
    target = return_np_array(plain_image, False, None)
    
    # get pixel values of class rgb source
    pix_image = return_np_array(class_file,False,None)
    # get bounding boxes of class source that crops the class
    pix_mask = return_np_array(class_file,True, cla)
    bbox = get_bounding_box(pix_mask)
    
    # check if the file is cropped at any corner
    if bbox[0] == 0 or bbox[1] == 480 or bbox[2] == 0 or bbox[3] == 480:
        print(f'{class_file}_{cla} is cropped')
        class_files.remove(class_file)
        continue
    
    # get cropped image of source using bounding boxes
    image_slice = get_img_slice(bbox, pix_image)
    mask_slice = get_img_slice(bbox, pix_mask, mask = True)
    
    # generate new data, keep padding always False, # dont get confused: this padding has no meaning - the padding
    # of the image slices happens using grey_erosion
    new_image, new_mask = generate_training_data(image_slice, mask_slice, target, bbox, padding = False)
    
    # save data to disk
    img = Image.fromarray(new_image)
    img.save(root + f'data/03_generated/images/{cla}/' + f'tile_{file}_lidar_generated.tif', 'TIFF')
    
    img = Image.fromarray((new_mask).astype('uint8'), mode='L')
    img.save(root + f'data/03_generated/masks/{cla}/' + f'tile_{file}_mask_{cla}_generated.tif', 'TIFF')
    
    

#%% show result
fig, axs = plt.subplots(nrows = 2, ncols = 2)
axs[0,0].imshow(return_np_array(plain_image, False, None))
axs[0,0].set_title('Target_Origin')

axs[0,1].imshow(image_slice)
axs[0,1].set_title('Class_Slice')

axs[1,0].imshow(new_image)
axs[1,0].set_title('Generated_Image')

axs[1,1].imshow(new_mask)
axs[1,1].set_title('Generated_Mask')

plt.tight_layout()

#%% 
"""
Remove trash images from 03/generated and then run the following code 
Trash images are those with class_slices in areas where they make no sense.
to automatically delet the corresponding masks
"""

remaining_imgs = os.listdir(root + f'data/03_generated/images/{cla}')
remaining_imgs_n = [f.split('_')[1] for f in remaining_imgs]

remaining_masks = os.listdir(root +  f'data/03_generated/masks/{cla}')
remaining_masks_n = [f.split('_')[1] for f in remaining_masks]

to_del = [d for d in remaining_masks_n if d not in remaining_imgs_n]
print(f'Found {len(to_del)} images to delete.')


to_del_paths = [root + f'data/03_generated/masks/{cla}/' + f'tile_{i}_mask_{cla}_generated.tif' for i in to_del]

for to_del in to_del_paths:
    os.remove(to_del)
print('Done')