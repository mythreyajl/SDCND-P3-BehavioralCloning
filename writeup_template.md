# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/model.png          "Model Visualization"
[image2]: ./images/center_driving.jpg "Center Driving"
[image3]: ./images/recovery1.jpg      "Recovery Image 1"
[image4]: ./images/recovery2.jpg      "Recovery Image 2"
[image5]: ./images/recovery3.jpg      "Recovery Image 3"
[image6]: ./images/center.jpg         "Center Image"
[image7]: ./images/left.jpg           "Left Image"
[image8]: ./images/right.jpg          "Right Image"
[image9]: ./images/fliplr.jpg         "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes and depths between 20 and 160. 

The model includes RELU layers to introduce nonlinearity, apart from MaxPooling2D to reduce network complexity. The data is normalized in the model using a Keras lambda layer.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an Adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used the data provided by Udacity as a baseline and augmented it with data that contained center lane driving, recovering from the left and right sides of the road especially at the bridge and the two sharp turns following the bridge. I also had several repeatitions of examples of driving the vehicle in the slower corners, duplicating the frames as needed. I also collected some data from track 2 to reduce overfitting to track 1.

### Architecture and Training Documentation

#### 1. Solution Design Approach

I started off with an initial goal of keeping the vehicle on the road for as long as possible. The network I used was very small with 2 convolutional layers. However, this network didn't learn to cope with slower, sharper turns. To this end, I tried increasing the dataset to include a lot of turns, a good way of doing that was to collect data on track 2. However, the network still didn't perform too well in the sharp turns. I mimiced the LeNet architecture which still wasn't big enough. So I was inspired by the Nvidia architecture and I added more convolutional layers to my solution. I split the data into training and validation set to help train on certain objectives. I noticed that there was a data insufficiency in the bridge and the two turns that followed it. To fix this issue, I collected recovery data that aimed to correct the vehicle's trajectory from where it was veering off-course. At the end, using the architecture that I have submitted, training it over 3 epochs, I came up with a model that autonomously drove around track 1 without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a cropping layer and a normalization lambda layer at the input. These were followed by the below 2D convolutional layers, each of which were activated by a ReLU for introducing non-linearities followed by a MaxPooling2D and a Dropout layer:
* Convolution2D (20 layers, 5x5 kernel)
* Convolution2D (40 layers, 5x5 kernel)
* Convolution2D (80 layers, 5x5 kernel)
* Convolution2D (120 layers, 5x5 kernel)

These were then flattened and followed by a series of fully connected layers with depths 400, 200, 120 and 1. The final quantity being regressed was the steering angle.

Below is a visualization of the network used:

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

I used the data provided for download as a good baseline to capture good driving behavior. Here's an example of center lane driving from the original dataset.

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to drive on the bridge and negotiate tight curves effectively. The following imates show the vehicle being trained to recover from points on the edge of the drivable surgace :

![alt text][image3]
![alt text][image4]
![alt text][image5]

I obtained some more data by driving on track 2 in the center lane.

To augment the data set, I used the center, left and right images correspoding to one frame. There was a correction factor applied to the left and right camera to adjust the steering angle according to the Point-of-View of the left and right cameras. I also used numpy to flip the image and flip the corresponding angle(s) for center, left and right cameras. Below are sample center, left, right and center-flipped images:

![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image9]

After the collection process, I had a little over 72000 data points. I then preprocessed this data by trimming to leave only a mask of 3(channels)x60(y-pixels)x320(x-pixels) per image

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by the increase of validation accuracy.  I used an adam optimizer so that manually training the learning rate wasn't necessary.
