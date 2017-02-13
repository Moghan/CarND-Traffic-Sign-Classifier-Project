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

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"
[custom_images]: ./examples/custom_images.png "Custom images"
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

####2. Model Architecture 

The network architecture is biult in code cell 8.

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

The training pipeline comes from the LeNet lab which means AdamOptimizer and softmax_cross_entropy_with_logits is used.
Epochs was set to 400 and learning rate to 0.0001.
Batch size was kept at(from LeNet-lab) 128.

 

####3. Training, validation and testing data.
20 % of the training data was split for validation. This was done in code cell 7 using the function sklearn.model_selection.train_test_split. The train_test_split function, split randomly and renders the remaining train dataset shuffled.

The test set was provided in the traffic-signs-data file and consisted of 12630 examples.



To cross validate my model, I randomly split the training data into a training set and validation set. I did this by ...

My final training set had X number of images. My validation set and test set had Y and Z number of images.

The sixth code cell of the IPython notebook contains the code for augmenting the data set. I decided to generate additional data because ... To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 




####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth cell of the ipython notebook. 

To train the model, I used an ....

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 
