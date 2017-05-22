# **Traffic Sign Recognition** 

## Writeup

---

[//]: # (Image References)

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/sample.png "Sample"
[image3]: ./examples/download.png "Downloaded images"
[image4]: ./examples/featuremaps1.png "Feature map 1"
[image5]: ./examples/featuremaps2.png "Feature map 2"
[image6]: ./examples/featuremaps3.png "Feature map 3"
[image7]: ./tensorboard.png "Tensorboard Graph"

## Files submited

    1. Traffic_Sign_Classifier.ipynb
    1. writeup.md
    1. Traffic_Sign_Classifier.html

---

### This is my project code 
[project code](https://github.com/sandyleo26/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### 1. Data Set Summary & Exploration

* Number of training examples = 34799
* Number of testing examples = 12630
* Image data shape = (32, 32)
* Number of classes = 43

### 2. Include an exploratory visualization of the dataset.

Here are some samples from the training set

![train sample][image2]

Here is an exploratory visualization of the data set. It is a bar chart showing sign distribution in train, validation and test set

![alt text][image1]

We can see that the distrubtions among 3 sets are generally the same. But within an individual distribution, some signs are more abundant than others. For example, we have much more sign 2 and 3 (30 & 50km/h) then 19 or 37 (Dangerous curve to the left & Go straight or left)

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

First, I decided to normalize the images using the simple formula provided, i.e. (pixel-128)/128. It was meant to be a starting point but turns out perform well.

After reading the [provided paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf), I think using simple preprocessing should be enough. Although it states that using grey scale achieve higher accuracy, the worst result is still better than 98%, which will well satisfy the requirement of this project. Same goes for augmented data set.

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x32 				    |
| Fully connected		| 800 x 120        							    |
| RELU					|												|
| Dropout				| keep prob = 0.5								|
| Fully connected		| 120 x 84        							    |
| RELU					|												|
| Fully connected		| 84 x 43        							    |
| Softmax&Cross Entropy	|												|
|						|												|

Here is the graph shown in Tensorboard

![Tensorboard Graph][image7]


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The hyperparameters I use are below:

    EPOCHS = 20
    BATCH_SIZE = 128
    rate = 0.001

I use the same batch size and learning rate from LeNet lab, which is 128 and 0.001 respectively. The number of EPOCHs is changed to 20 due because the training accuracy still has potential to improve (0.984). 

The optimizer used, AdamOptimizer, is again inherited from LeNet lab. Posteri [research](https://arxiv.org/abs/1412.6980) reveal that AdamOptimizer, a relatively new(2014) momentum based optimizer, has better converging speed compared with [RMSDrop and AdaGrad](https://moodle2.cs.huji.ac.il/nu15/pluginfile.php/316969/mod_resource/content/1/adam_pres.pdf). It maintains exponential moving averages of gradient and its square. It combines the advantages of AdaGrad and RMSDrop and widely used in deep learning community (e.g. Google DeepMind).

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* Training Accuracy = 0.992
* Validation Acurracy = 0.952
* Test Accuracy = 0.842

If an iterative approach was chosen:

* What was the first architecture that was tried and why was it chosen?
    The first architecture started with about 2 times of parameters in the 2 convolutional layers and without dropout. This is chosen because compared with LeNet, this dataset has 3 channels input thus should have more parameters to train

* What were some problems with the initial architecture?
    The validation accuracy is much lower than training accuracy which is close to 1. It means overfitting.

* How was the architecture adjusted and why was it adjusted?
    The filter sizes are reduced and a dropout layer is added. There's still overfitting but now validation accurary is close to 95%

###Test a Model on New Images

Here are nine German traffic signs that I found on the web:

![sample images downloaded][image3]

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

    1. 4 Speed limit (70km/h)
    1. 25 Road work
    1. 12 Priority road
    1. 34 Turn left ahead
    1. 3 Speed limit (60km/h)
    1. 13 Yield
    1. 35 Ahead only
    1. 14 Stop
    1. 17 No entry
    1. 12 Priority road

The model was able to correctly guess 9 of the 10 traffic signs, which gives an accuracy of 90%. This compares poorly to the accuracy on the test set of 94.9%.

The first 9 images are downloaded from GTSDB sites and are all 80x80 in png format. So down-sampling is needed to match the inputs requirements of our neural network. The images are better compared with samples of train set above, in terms of contrast, distortion, noise and sharpness. So it's expected the prediction will be on par with, in not better, than the training accuracy.
The last image, which is supposed to be a "No stopping" sign, is not in the training set. It'll be interesting to see how model predict given an sign it never learns.

####The submission includes five new German Traffic signs found on the web, and the images are visualized. Discussion is made as to particular qualities of the images or traffic signs in the images that are of interest, such as whether they would be difficult for the model to classify

The total prediction accuracy is 90%. The prediction for the first 9 images are all correct as expected. But for some images, the confidence of the prediction is not as high as others. For example, for image 9, "No entry", the first guess is 60% while for image 5, "60km/h" the first and second guess are 17% and 12% ("80km/h") respectively. This is due to the similarities of the candidates. For "No entry" sign, it can't find a candidate that matches it easily but one can argue that all the speed limit signs are similar to each other.

As for image 10, the result is more interesting. The first choice is "Priority road" (image 3) which is hardly close to the sign's blue background with red cross and circle. 3 possible reasons for this are

* Priority road is one of the most common signs in training set which distort the model
* the yellow diamond shape are somewhat related to the red cross 
* the model is just not well trained

The next 4 guesses are probabibly due to they all have a blue background with thick lines in the center.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code and result can be found in the exported [html](./Traffic_Sign_Classifier.html) or jupyter [notebook](./Traffic_Sign_Classifier.ipynb). But here is an interest finding:

For the 1st image (70km/h), the top 5 are:


We can see that the model gives relatively low probability to all guesses although the best guess is still twice as the runner up

For the 9th image (No entry), the top 5 are:


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

'No Entry' (1st)
![alt text][image4]

'Ahead Only' (6th)
![alt text][image5]

'Road Work' (7th)
![alt text][image6]

We can see that different feature maps capture different characteristics. One capture the outer shape, some captures separate parts of inner sign, some captures luminance differences.
