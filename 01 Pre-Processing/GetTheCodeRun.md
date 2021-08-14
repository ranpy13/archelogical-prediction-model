# How can I get the code run?

We realize that many parts of the code are redundant and could be shortened. However, in the short time available and as an after-work project, this was not always feasible. We are therefore open to help to get the code running.
 - The data to read in are stored from the root directory under `\data\01_raw\lidar_train` and `\data\01_raw\train_masks`.
 - The generated images and masks will be safed in `\data\03_generated\images` and `\data\03_generated\masks`, respectively.

There are three files with speaking names that generate the different kinds of new training data. The `.py`-files are stored in `\src\d03_processing`. Note that due to the random placing and random selection of the data, the training data generated will be different each time you run the code. We did not use any `random_state = 0` here.
 - `generate_data_padded.py` creates the padded data. The size of the paddings can be varied under the keyword argument `size=(n,n)` for the function `grey_erosion`.
 - `generate_data_pixel_precise.py` creates the pixel-precise cropped images. During the learning process, we first starting selecting manually class_images that might be more representative for the ground distribution. The problem here was that we did not want edge-cutted training images anywhere in the new generated image where the edge cut doesn`t make sense. In this version provided here however, we simply took all class_images that might be worth copying into an empty image as described in the documentation.
 - `generate_data_roughly_cropped` creates the **rectangled** croped images.

### You can directly contact me if the code doesn't run as expected: johannes.allgaier@uni-wuerzburg.de.