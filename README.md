# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This repository contains files for the Behavioral Cloning Project
which demonstrates the power of deep neural networks and convolutional neural networks to clone driving behavior. 

Keras was used to train, validate and test an optimal driving model. The model will output a steering angle to an autonomous vehicle.

Structure
* `model.py` (script used to create and train the model)
* `utils.py` (script used to support model.py)
* `drive.py` (script to drive the car)
* `model.h5` (a trained Keras model)
* `final.mp4` (a video recording of your vehicle driving autonomously around the track for at least one full lap)

Credit to the Udacity team for the resources. Probably 

Preprocessing
---
### 1) Image processing pipeline

Inspired from [@naokishibuya](https://github.com/naokishibuya/car-behavioral-cloning), the processing pipeline supports the model definition and adds
diversity to the training dataset. The result is detailed below

* Center: the main camera frame angle
* Left:
* Right: 
* Cropped: this step eliminates the sky and car hood out of the camera frame. These noises are not helpful to guide the car on track.
* Random Flip:
* Random Translate:
* Random Brightness:
* Random Shadow

![2973](figs/fig_2973.png)
![4810](figs/fig_4810.png)

### 2) Generator

``` python
# Generator function
```

### 3) Collect additional data

I drove the track twice, one in anti-clockwise and turned around to go clockwise. This adds more diversity to the camera angles to be captured. One of the main tricks to teach the car to drive well is recovery from a hard or bad turns. In my additional data collection, I intentionally made bad turns and recover with sharp steers m


Train
---
### 1) Model definition

Below is the model parameters summarized by Keras `model.summary()`. This is based on the [Nvidia model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) that has been successful in self-driving tests. There are 252,219 weights parameters to train.

Layer types:
* Convolution layers:
* Dropout layer:
* Fully-connected layers:

![model](figs/nvidia.png)

### 2) Training operations

A few training features were used

* Adam optimizer
* Keras callbacks
* `fit_generator()`

### 3) Hyperparameters

* Correction factor for left/right driving images
* Learning rate and decay rate


Result
---
Observations:
* Reduction in validation loss does not translate to better autonomous driving. Because of that, callbacks need to be set for every single epoch.
The later epoch tends to have better performance because the training set might have been shuffled to difficult turns.


### 1) [Early Epoch Video](vid/early.mp4)

As shown, the car could not make the first turn and end up on the grass.

![crash](figs/early.jpg)

### 2) [Mid Epoch Video](vid/mid.mp4)

As shown, the car was able to complete about 70% of the track, but ended up in the river. Possible explanation is that the driving sample for that
particular segment might have not been shown to the model.

![in_river](figs/in_river.jpg)

### 3) [Late/Final Epoch Video](final.mp4)

As shown, the car completed the track and final video is in `vid/final.mp4`

![smooth](figs/smooth.jpg)

