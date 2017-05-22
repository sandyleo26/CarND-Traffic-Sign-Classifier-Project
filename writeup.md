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

We can see that the distributions among 3 sets are generally the same. But within an individual distribution, some signs are more abundant than others. For example, we have much more sign 2 and 3 (30 & 50km/h) then 19 or 37 (Dangerous curve to the left & Go straight or left)


#### Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

First, I decided to normalize the images using the simple formula provided, i.e. (pixel-128)/128. It was meant to be a starting point but turns out perform well.

After reading the [provided paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf), I think using simple preprocessing should be enough. Although it states that using grey scale achieve higher accuracy, the worst result is still better than 98%, which will well satisfy the requirement of this project. Same goes for augmented data set.

#### Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

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


#### Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The hyperparameters I use are below:

    EPOCHS = 20
    BATCH_SIZE = 128
    rate = 0.001

I use the same batch size and learning rate from LeNet lab, which is 128 and 0.001 respectively. The number of EPOCHs is changed to 20 due because the training accuracy still has potential to improve (0.984). 

The optimizer used, AdamOptimizer, is again inherited from LeNet lab. Posteri [research](https://arxiv.org/abs/1412.6980) reveal that AdamOptimizer, a relatively new(2014) momentum based optimizer, has better converging speed compared with [RMSDrop and AdaGrad](https://moodle2.cs.huji.ac.il/nu15/pluginfile.php/316969/mod_resource/content/1/adam_pres.pdf). It maintains exponential moving averages of gradient and its square. It combines the advantages of AdaGrad and RMSDrop and widely used in deep learning community (e.g. Google DeepMind).

#### Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* Training Accuracy = 0.995
* Validation Acurracy = 0.957
* Test Accuracy = 0.947

If an iterative approach was chosen:

* What was the first architecture that was tried and why was it chosen?
    The first architecture started with about 2 times of parameters in the 2 convolutional layers and without dropout. This is chosen because compared with LeNet, this dataset has 3 channels input thus should have more parameters to train

* What were some problems with the initial architecture?
    The validation accuracy is much lower than training accuracy which is close to 1. It means overfitting.

* How was the architecture adjusted and why was it adjusted?
    The filter sizes are reduced and a dropout layer is added. There's still overfitting but now validation accurary is close to 96%

### Test a Model on New Images

Here are nine German traffic signs that I found on the web:

![sample images downloaded][image3]

#### The submission includes five new German Traffic signs found on the web, and the images are visualized. Discussion is made as to particular qualities of the images or traffic signs in the images that are of interest, such as whether they would be difficult for the model to classify

Based on human eyes and common sense, 

    1. the 1st (70km/h) and 2nd (yield) would be hard to predict due to its low brightness.
    1. the 4th (straigh or left), 5th (turn right) and 6th (roundabout) would be hard due to their abnormal aspect ratio as well as low quality.
    1. the 6th (wild animal crossing) would be hard duo to low resolution and similarities in the middle.

#### Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the actual results of the prediction:

    0. 4 Speed limit (70km/h)           - correct
    0. 34 Turn left ahead               - correct
    0. 13 Yield                         - correct
    0. 37 Go straight or left           - correct
    0. 34 Turn left ahead               - wrong
    0. 40 Roundabout mandatory          - correct
    0. 20 Dangerous curve to the right  - wrong

The accuracy is 5/7 ~= 71%, which is lower than test accuracy (>94%). It's probably understanable that these images are biased since they were picked due to their abnormality (e.g. distortion, low quality). However, if we can argument our training data such that these scenarios would be covered, the result would have been better.

#### Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code and result can be found in the exported [html](./Traffic_Sign_Classifier.html) or jupyter [notebook](./Traffic_Sign_Classifier.ipynb).

I'm a bit surprised to find that my model is very certain in most of the cases, only image 4, 5 and 7 give more than 1 non-zero probabilities.

For image 4, the best guess is correct with significant margin than other guesses. The possibles reasons why the other 2 candidates are chosen are probably due to their similarities and the low quality of the downloaded image.

    0. Go straight or left(37) - 0.67
    1. Keep left(39) - 0.19
    2. Keep right(38) - 0.14

For image 5, the best guess is obviously wrong. Actually, it gives the exact opposite answer yet the correct answer only rank 4th with 1% possibility. It might be helpful to visualize the internal state when test image is feed in to the network.

    0. Turn left ahead(34) - 0.93
    1. Roundabout mandatory(40) - 0.03
    2. Ahead only(35) - 0.03
    3. Turn right ahead(33) - 0.01

For the last image, the guesses (especially the top guess) show a lot similarity with the test image but due to low quality of the image, the model failed to predict correctly.

    0. Dangerous curve to the right(20) - 0.92
    1. Children crossing(28) - 0.05
    2. End of no passing(41) - 0.01
    3. Slippery road(23) - 0.01
    4. Beware of ice/snow(30) - 0.01


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

'No Entry' (1st)
![alt text][image4]

'Ahead Only' (6th)
![alt text][image5]

'Road Work' (7th)
![alt text][image6]

We can see that different feature maps capture different characteristics. One capture the outer shape, some captures separate parts of inner sign, some captures luminance differences.
