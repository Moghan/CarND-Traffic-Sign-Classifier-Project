#**Traffic Sign Recognition** 
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[class5_images]: ./examples/class_5.png "Class 5 images"
[class10_images]: ./examples/class_10.png "Class 10 images"
[class15_images]: ./examples/class_15.png "Class 15 images"
[class20_images]: ./examples/class_20.png "Class 20 images"
[custom_images]: ./examples/custom_images.png "Custom images"
[custom_images_gray]: ./examples/custom_images_gray.png "Custom images"
[orig_images]: ./examples/orig_train_images.png "Original training images"
[prep_images]: ./examples/prep_train_images.png "Preprocessed training images"
[class_count]: ./examples/class_count.png "Number of examples of each class"

## Rubric Points

---
###Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/Moghan/CarND-Traffic-Sign-Classifier-Project/edit/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the python library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 39209
* The size of test set is 12630
* The shape of a traffic sign image is (32x32x3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third and fourth code cell of the IPython notebook.  

As we can see in this chart there is big differences between number of examples among the classes. My guess is that this leads to a network that is better at recognizing signs that had many training examples in their class, compared to 
signs with fewer examples. Balancing throu augmentation may be a future improvement. On the other hand, maybe it is more important to get the most common signs right.

![alt text][class_count]

Example of images from a some of the classes:
![alt text][class5_images]
![alt text][class10_images]
![alt text][class15_images]
![alt text][class20_images]

###Design and Test a Model Architecture

#### 1. Preprocess image data
Preprocess of image data is done in the 5th code cell.

* First - Grayscale 

I read an [article](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) by Pierre Sermanet and Yann LeCun that said they got better results ignoring color information. So I made grayscaling a priority.

* Second - Normalization

Vincent Vanhoucke was in lecture very clear that normalization is a good idea. My (x-128)/128 made awful images and bad results when training. Decided for a x/255 formula instead, and values between 0 and 1.


Example of traffic sign images before and after preprocess.

![alt text][orig_images]
![alt text][prep_images]

An improvement  would be to increase the number of training examples by augmentation. I would start with increasing the number 400%.

####2. Model Architecture 

The network architecture is built in code cell 8.

During lectures I got curious on inception, so I started with an architecture with two inception layers, two CNN and two fully connected layers. It took more than 2 minutes / epoch to train and made 97% on training.

Than I read [the article](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) by Sermanet and LeCun, which have inspired a lot in this project.

Started over and come to this:


| Layer               		|     Description	        					|     Multi-scale points | 
|:---------------------:|:---------------------------------------------:| :---------------------|
| Input               		| 32x32x1 image (gray)                         	|                       |
| Convolution 3x3      	| 1x1 stride, same padding, outputs 32x32x16   	|                       |
| RELU				             	|										                                   		|                       |
| Convolution 3x3      	| 1x1 stride, same padding, outputs 32x32x16   	|                       |
| RELU					|												|                                               |
| Max pooling	         	| 2x2 stride,  same padding, outputs 16x16x16  	|---->  MS1             |
| Convolution 3x3      	| 1x1 stride, same padding, outputs 16x16x32   	|                       |
| RELU					|												|                                                |
| Convolution 3x3      	| 1x1 stride, same padding, outputs 16x16x64 	  |                       |
| RELU					|												|                                               |
| Max pooling	         	| 2x2 stride,  outputs 8x8x64 				              | ---->  MS2            |
| Flatten MS1 and MS2   | flatten the result from prev pooling ops		    |                       |
| Concat MS One and Two | multi-scale with MS1      					               | <----  MS1 + MS2      |
| Fully connected     		| Input 8192 , Output 120        				           ||
| Fully connected	     	| Input 120 , Output 43        					            ||



 

####3. Training, validation and testing data.
20 % of the training data was split for validation. This was done in code cell 7 using the function sklearn.model_selection.train_test_split. The train_test_split function, split randomly and renders the remaining train dataset shuffled.

This led to a validation set of 7842 examples.

The test set was provided in the traffic-signs-data file and consisted of 12630 examples.



####4. Model training
The training pipeline comes from the LeNet lab which means AdamOptimizer and softmax_cross_entropy_with_logits is used.
Epochs was set to 400 and learning rate to 0.0001.
Batch size was kept at(from LeNet-lab) 128.

The code for training the model is located in the tenth cell of the ipython notebook. 

####5. Solution design
With the LeNet project as a starting ground, I went straight for the architecture, and training on different NN designs. Pretty much trial and error, much fun and rewarding. Wanted to try inception and looked at schematics of AlexNet and GoogleNet for inspiration. After a while I had a working network, but very slow to train and the resulting accuracy was not more than 97% while training. Then I read a couple of articles on Convnets.

The ideas for my final solution comes from [Traffic Sign Recognition with Multi-Scale Convolutional Networks](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) and [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/pdf/1512.00567.pdf).
These made me ditch inception, go for double ConvLayer(3x3) instead of single ConvLayer(5x5), and use multi-scaling. I also decided for grayscaling.
The increasing the number of filters was also part of the design. It goes from 16, 16, 32 to 64 in convnets, before flattened and fully connected layers take over.

I iterated(trial and error) to find learning rate and epochs. I would like to implement a decreasing learning rate and a automated epoch stop, but time is running out.

I am satisfied with the architecture for now. I think improvements is best found working with the data sets.


The code for calculating the accuracy of the model is located in the elevent cell of the Ipython notebook.

My final model results were:
* training set accuracy of ?. I never measured this accuracy.
* validation set accuracy of 99.5%
* test set accuracy of 95.6%


###Test a Model on New Images

Here are the eleven German traffic signs that I found on the web:

![alt text][custom_images]
and after preprocess
![alt text][custom_images_gray]
Compared to some training examples, the accuracy should be high.
![alt text][orig_images]

I must confess that these custom images was normalized by default. Therefor they have only been converted to grayscale, and therefor have not passed the same preprocess pipeline as the others sets. 

Prediction of custom images:
100 % accuracy sounds good, but the pictures was clean and without noise. 
The three lowest (winning) prediction probabilities was 92%, 94% and 99%. 

I suppose this...compares favorably to the accuracy on the test set of 95.6 %


The code for making predictions on my final model is located in the 12th cell of the Ipython notebook.



####3. Softmax probabilities for custom images
The code for making predictions on my final model is located in the 13th cell of the Ipython notebook.

I show the probabilty for the first three images. (They all look pretty much the same.)

![alt text][custom_images]
First image:
* 0.92 percent probabilty of :Right-of-way at the next intersection
* 0.08 percent probabilty of :General caution
* 0.00 percent probabilty of :Roundabout mandatory
* 0.00 percent probabilty of :Wild animals crossing
* 0.00 percent probabilty of :Dangerous curve to the right

Second image:
* 1.00 percent probabilty of :Road work
* 0.00 percent probabilty of :Road narrows on the right
* 0.00 percent probabilty of :Double curve
* 0.00 percent probabilty of :Right-of-way at the next intersection
* 0.00 percent probabilty of :Turn right ahead

Third image:
* 1.00 percent probabilty of :Keep right
* 0.00 percent probabilty of :Speed limit (30km/h)
* 0.00 percent probabilty of :Roundabout mandatory
* 0.00 percent probabilty of :Slippery road
* 0.00 percent probabilty of :Speed limit (20km/h)
