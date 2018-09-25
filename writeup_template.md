# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/web1.png "Traffic Sign 1"
[image5]: ./examples/web2.png "Traffic Sign 2"
[image6]: ./examples/web3.png "Traffic Sign 3"
[image7]: ./examples/web4.png "Traffic Sign 4"
[image8]: ./examples/web5.png "Traffic Sign 5"
[image9]: ./examples/probs.png "web images predict probs"
[image10]: ./examples/nn_vis1.png "nn first layer feature maps"
[image11]: ./examples/nn_vis2.png "nn first layer feature maps"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data distributed across the 43 classes.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)


It's believed to be benefitial to normalize the input image by manually calculate the mean and stdev of the dataset, then normalize them by `(x - mean) / std`. However I'll use a innovative method, eliminating this additional technique.

The idea is based on the deep understanding of Batch Normalization. BN is widely used to fight with the "covariance shift" before each activation layer. However if we take a closer look, the input data can also be viewed as a layer of activation, and applying BN to it equivalently normalize the input data. Overtime, BN will estimate the mean and std of the dataset and apply the same `(x - mean) / std` operation on it. What's more, BN also have learnable offset and scale, where Back Propagation will decide how to use them.

In conclusion, using BN for input data is superior to the manual normalization approach. So I'll skip manual normalization here and add a BN as first layer in the model instead.

I decided not to augment data because I added a lot of regularizations to my model, and they already works very well.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| BN             		| 32x32x3 RGB image normalized					| 
| Convolution 3x3     	| 1x1 stride, SAME padding, outputs 32x32x16 	|
| BN             		| normalized activations     					| 
| RELU					| 												|
| DropOut				| regularization								|
| Convolution 3x3     	| 2x2 stride, SAME padding, outputs 16x16x32  	|
| BN             		| normalized activations     					|
| RELU					|												|
| DropOut				| regularization								|
| Convolution 3x3     	| 2x2 stride, SAME padding, outputs 8x8x64  	|
| BN             		| normalized activations     					|
| RELU					|												|
| DropOut				| regularization								|
| Convolution 3x3     	| 2x2 stride, SAME padding, outputs 4x4x128  	|
| BN             		| normalized activations     					|
| RELU					|												|
| DropOut				| regularization								|
| Flatten				| outputs 2048									|
| FC1           		| Fully Connected outputs 512 					|
| BN             		| normalized activations     					|
| RELU					|												|
| DropOut				| regularization								|
| FC2           		| Fully Connected outputs 128 					|
| BN             		| normalized activations     					|
| RELU					|												|
| DropOut				| regularization								|
| FC3           		| Fully Connected outputs 43 					|
| BN             		| normalized activations     					|
| RELU					|												|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam Optimizer to reduce softmax loss. Batch size 128, 20 epoches and learning rate 0.001.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.998
* validation set accuracy of 0.975
* test set accuracy of 0.968

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen? Same as MNIST.
* What were some problems with the initial architecture? Accuracy not hight enough.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
  * Add 2 additional conv layers -> Fight underfitting by adding more params
  * Add number of feature maps for each conv layers and FC layer -> Fight underfitting by adding more params
  * Add BatchNormalization and DropOut layers -> Fight overfitting. Add regularization
  * Replace MaxPool with Conv with stride 2 to save computation
* Which parameters were tuned? How were they adjusted and why?
  * number of feature maps and dropping rate of DropOut layers
  * First try to overfit (val loss high, train loss low), then increase dropping rate. (val loss low, train loss loss)
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
  * Replace MaxPool with fully convolutional layers, so that we save computation
  * Add BN and Dropout, so that the model generalize well
  * Use BN as first layer to normalize input data, so that we get "learned" normalization of input data

If a well known architecture was chosen: NO
* What architecture was chosen? N/A
* Why did you believe it would be relevant to the traffic sign application? N/A
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well? N/A
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| Road work     		| Road work 									|
| Priority road			| Priority road									|
| No vehicles	      	| No vehicles					 				|
| Speed limit (120km/h)	| Speed limit (120km/h)     					|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 96.8%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the "Predict the Sign Type for Each Image" cell of the Ipython notebook.

![alt text][image9]

For the first image, the model is very sure that this is a stop sign (probability of 0.999353), and the image does contain a stop sign.

For the forth image, the model is not very sure. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.73906106         			| No vehicles   						| 
| 0.12204006     				| Yield 								|
| 0.027229566					| Speed limit (30km/h)					|
| 0.010470875	      			| No passing					 		|
| 0.0080910679				    | Priority road      					|


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

- Edges, as you can see in the first example, FeatureMap 0
- Colors, as you can see in the second example, FeatureMap 5
![alt text][image10]
![alt text][image11]
