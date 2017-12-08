# **Behavioral Cloning of a Simulated Driving Course** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/center_lane.jpg "Center Lane Image"
[image2]: ./examples/off_road_1.jpg "Off Road Correction"
[image3]: ./examples/off_road_2.jpg "Off Road Correction"
[image4]: ./examples/off_road_3.jpg "Off Road Correction"
[image5]: ./examples/off_road_4.jpg "Off Road Correction"
[image6]: ./examples/normal_image.jpg "Normal Image"
[image7]: ./examples/flipped_image.jpg "Flipped Image"
[image8]: ./examples/histogram_initial.png "Initial Distribution"
[image9]: ./examples/histogram_33.png "Equally Random Distribution" 
[image10]: ./examples/histogram_45_1.png "10/45/45 Distribution 0.2 Correction" 
[image11]: ./examples/histogram_45_2.png "10/45/45 Distribution 0.25 Correction" 
[image12]: ./examples/histogram_45_3.png "10/45/45 Distribution Mouse Precision"
[image13]: ./examples/autonomous_run.gif "Fully Autonomous Run"

---
### Files & Code Quality

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 1. Executing autonomous run from trained model.

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 2. Training autonomous model.

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. Training can be carried out by executing
```sh
python model.py
```
Be sure to capture training data and and update the file names according to the comments found in model.py.


### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network beginning with 3x3 filter sizes then 2x2's. The filter depths range between 24 and 128 (model.py lines 95-101) 
The model includes RELU layers to introduce nonlinearity (code line 95,98-101), and the data is normalized in the model using a Keras lambda layer (code line 94).

#### 2. Attempts to reduce overfitting in the model

The model contains a dropout layer in order to reduce overfitting (model.py lines 96). 

I introduced a single dropout layer after the first RELU in other to avoid overfitting my sample data, especially since it was relatively difficult to continously gather new data.

The model was trained and validated on different data sets (70/30 split) to ensure that the model was not overfitting (code line 60 - 62, 109). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 108).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. I also drove a few laps in the oposite direction in order to avoid a bias to the original track in the sample data. 

For details about how I created the training data, see section 3 of the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to gather the right training sample data, so that the model knows the right things to focus on. Another apporach was to base the eventual model off a previously existing model that has demonstrated good performance on a similar problem.

My first step was to use a convolution neural network model similar to the architecture published by the autonomous-vehicle team at NVIDIA. I thought this model might be appropriate because the team at NVIDIA used this architecture to successfully train their automous vehicles.

I started off by croping the input to only show our area of interest. This was the surounding trees, rocks and objects of no importance had less of an impact on the training.
Similar to the NVIDIA architecture, I introduced a normalization layer to normalize the number range (-0.5 to 0.5) of the input images.
This was followed by 5 convolutional layers with filter sizes ranging from 3x3 and 2x2. The filter had depths ranging from 24 to 120.
This was then followed by 4 fully connected layers similar to the NVIDIA architecture.

I applied some down-sampling to improve the learning speed of the model by introducing a Max Pooling layer after the dropout layer.

To combat the overfitting, I modified the model by introducing a dropout layer after the first convolutional layer. I applied a 25% dropout rate, which according to the vaildation loss results was enough to avoid overfitting. 

After creating the model, I conducted training using an Adam Optimizer to I need not need to fiddle with the learning rate. The final model ran for 8 epochs before a plateau on the loss improvement occured.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track especially around sharp bends or areas that had close hills present. To improve the driving behavior in these cases, I collected a lot more showing how to correctly take those sharp bends. I collected this data both in the normal direction and in the oposite direction. As for the issues with the hills, I solved that by cropping the camera image to focus on the course and ignore objects that shouldn't be considered.

At the end of the process, the vehicle is able to drive autonomously around the track multiple times without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 92-106) consisted of a convolution neural network with the following layers and layer sizes:

| Layer                 | Description                                   | 
|:---------------------:|:---------------------------------------------:| 
| Input                 | 160x320x3 BGR image                           | 
| Image Cropping        | 60px(top) and 20px(bottom), outputs 80x320x3  | 
| Normalization         | Convert (0 - 255) tensors to (-0.5 - 0.5)     |
| Convolution 3x3       | 2x2 stride, valid padding, outputs 40x160x24  |
| RELU                  |                                               |
| Dropout               | Droupout rate 0.25                            |
| Max pooling 2x2       | 2x2 stride, valid padding, outputs 20x80x24   |
| Convolution 3x3       | 2x2 stride, valid padding, outputs 10x40x36   |
| RELU                  |                                               |
| Convolution 3x3       | 2x2 stride, valid padding, outputs 5x20x48    |
| RELU                  |                                               |
| Convolution 2x2       | 1x1 stride, valid padding, outputs 4x19x64    |
| RELU                  |                                               |
| Convolution 2x2       | 1x1 stride, valid padding, outputs 3x18x120   |
| RELU                  |                                               |
| Flatten               | outputs 6480                                  |
| Fully connected       | outputs 100                                   |
| Fully connected       | outputs 50                                    |
| Fully connected       | outputs 10                                    |
| Fully connected       | outputs 1                                     |
|                       |                                               |

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to get back unto the track if it ever went off course.
These images show what a recovery looks like starting from being off to the right of the track to the center of the track :

![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process by driving in the oposite direction in order to get more data points.

To augment the data set, I also flipped images and angles thinking that this would help avoid a left turn bias, since most of the track veers left. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

After the collection process, I had 13457 number of data points. By creating a histogram of the steering angle distribution for the sample data I could see that there was a heavy bias towards keeping the steering straight.

![alt text][image8]

Training with this data resulted in a 0.03 loss which seemed good, but hid the implicit straight bias. When an autonomous run was simulated with the trained model, the car veered off the road immediately because it didn't adequately steer.

I then preprocessed the sample data by randomly using the left and right camera images as a center image and applying a fake steering angle correction to simulate steering.
This significantly increased the number of non-zero steering angles. I started with a split of center camera (33%), left camera (33%), right camera (33%). I applied a 0.2 angle of correction to the left and right camera images.

![alt text][image9]

The loss increased from 0.03 to 0.05 but the model performed better. The vehicle stayed on course for the first part of ride before encountering a sharp bend and veering off road.

I updated the data split with center camera (10%), left camera (45%), right camera (45%). This de-emphasized the importance of the straight(zero-steering angle) data. Therefore the model became a lot better at making turns.

![alt text][image10]

The loss dropped from 0.05 to 0.04 and the model performed better. The vehicle mostly stayed on course but slowly veered off after under-correcting at a bend.

I updated the angle of correction of the left and right camera images to 0.25 from 0.2.

![alt text][image11]

This model performed better at a loss of 0.05 but still veered off a few bends by slightly overcorrecting.

I decided to tackle the zero steering problem another way by using the mouse to steer instead of keyboard keys. This helped the model understand that steering correction is always happening at every stage of a ride. As a result there was much less zero steering angles and less extreme steering angle outliers.

![alt text][image12]

With a loss of 0.02, the vehicle successfully autonomously drove around track multiple times with this model.

This is a snippet of the a successful autonomous run by the trained model:

![alt text][image13]
