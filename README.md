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
1) Basic processing pipeline
2) Augmentation pipeline

3) Collect additional data

I drove the track twice, one in anti-clockwise and turned around to go clockwise. This adds more diversity to the camera angles to be captured. One of the main tricks to teach the car to drive well is recovery from a hard or bad turns. In my additional data collection, I intentionally made bad turns and recover with sharp steers m


Train
---
1) Model definition
2) Training operations
3) Hyperparameters


Result
---
1) Preliminary
2) After augmentation
3) Final