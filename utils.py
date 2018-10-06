import os, sys, glob, time

import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 160, 320, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
IMAGE_COLS = ('center', 'left', 'right')
STEER_COLS = ('steering')

    
def load_image(data_dir, image_file):
    """Load RGB images from a file"""
    
    return mpimg.imread(os.path.join(data_dir, image_file.strip()))


def resize(image):
    """Resize image to determined sizes"""
    
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)


def crop(image):
    """Crop the image (removing the sky at the top and the car front at the bottom)
    
    Credit: https://github.com/naokishibuya/car-behavioral-cloning
    """
    
    return image[60:-25, :, :] # remove the sky and the car front


def rgb2yuv(image):
    """Convert the image from RGB to YUV to support the NVIDIA model.
    
    Credit: https://github.com/naokishibuya/car-behavioral-cloning
    """
    
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


def random_flip(image, steer_angle, **config):
    """
    Randomly flip the image left <-> right, and adjust the steering angle.
    Credit: https://github.com/naokishibuya/car-behavioral-cloning
    """
    
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steer_angle = -steer_angle
        
    return image, steer_angle


def random_translate(image, steer_angle, **config):
    """
    Randomly shift the image vertically and horizontally (translation).
    Credit: https://github.com/naokishibuya/car-behavioral-cloning
    """
    
    range_x, range_y = config.get('range_x', 100), config.get('range_y', 10)
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steer_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    
    return image, steer_angle


def random_shadow(image, **config):
    """Generates and adds random shadow
    
    Credit: https://github.com/naokishibuya/car-behavioral-cloning
    """
    
    # (x1, y1) and (x2, y2) forms a line
    # xm, ym gives all the locations of the image
    x1, y1 = IMAGE_WIDTH * np.random.rand(), 0
    x2, y2 = IMAGE_WIDTH * np.random.rand(), IMAGE_HEIGHT
    xm, ym = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH]

    # Mathematically speaking, we want to set 1 below the line and zero otherwise
    # Our coordinate is up side down.  So, the above the line: 
    # (ym-y1)/(xm-x1) > (y2-y1)/(x2-x1)
    # as x2 == x1 causes zero-division problem, we'll write it in the below form:
    # (ym-y1)*(x2-x1) - (y2-y1)*(xm-x1) > 0
    mask = np.zeros_like(image[:, :, 1])
    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

    # Choose which side should have shadow and adjust saturation
    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)

    # Adjust Saturation in HLS(Hue, Light, Saturation)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)


def random_brightness(image, **config):
    """Randomly adjust brightness of the image.
    
    Credit: https://github.com/naokishibuya/car-behavioral-cloning
    """
    
    # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:, :, 2] =  hsv[:, :, 2] * ratio # Increase V for bright, only 50% at the time
    
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def process_image(image):
    """Create standard image preprocessing pipeline, with option to augment"""
    
    image = crop(image)
    image = resize(image)
    image = rgb2yuv(image)
    
    return image


def choose_adjust(data_dir, data_row, steer_angle, **config):
    """Choose randomly out of center/left/right and adjust steer
    
    Left = positive steer
    Right = negative steer
    """
    
    choice = np.random.choice(3)
    if choice == 0: # Center
        return load_image(data_dir, data_row[IMAGE_COLS[choice]]), steer_angle
    elif choice == 1: # Left, steer more left
        return load_image(data_dir, data_row[IMAGE_COLS[choice]]), steer_angle + config.get('steer_corr', 0.2)
    else: # Right, steer more right
        return load_image(data_dir, data_row[IMAGE_COLS[choice]]), steer_angle - config.get('steer_corr', 0.2)


def image_pipeline(data_dir, data_row, steer_angle, augment=True, **config):
    """Create random image augmentation for training"""
    
    image, steer_angle = choose_adjust(data_dir, data_row, steer_angle, **config)
    if augment:
        image, steer_angle = random_flip(image, steer_angle, **config)
        image, steer_angle = random_translate(image, steer_angle, **config)
        image = random_shadow(image, **config)
        image = random_brightness(image, **config)
    image = process_image(image) # Standard
    
    return image, steer_angle


def batch_generator(data_dir, image_df, steering_df, is_training=True, **config):
    """Generate batch to load. 
    
    A generator is like a coroutine, a process that can run separately from another main routine. More memory efficient
    
    Args:
        data_dir
        image_df: (n, 3) dataframe of ('center', 'left', 'right')
        steering_df: (n, 1) dataframe of ('steering)
        batch_size
        is_training
    
    Returns:
        generator
    """
    
    batch_size = config.get('batch_size', 64)
    
    while 1:
        feats = np.zeros((batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
        responses = np.zeros((batch_size))
        
        batch_image = image_df.sample(batch_size)
        batch_steer = steering_df[batch_image.index].reset_index(drop=True)
        batch_image.reset_index(inplace=True, drop=True)
        
        for i, row in batch_image.iterrows():
            if is_training:
                feats[i, :, :, :], responses[i] = image_pipeline(data_dir, row, batch_steer[i], augment=True, **config)
            else:
                feats[i, :, :, :], responses[i] = image_pipeline(data_dir, row, batch_steer[i], augment=False, **config)
                
        yield feats, responses