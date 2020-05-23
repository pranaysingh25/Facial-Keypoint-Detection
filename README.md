### In this project I have built a Facial Keypoints Detection System using a Convolutional Neural Network(CNN) that takes in any image with faces, and predicts the location of 68 distinguishing Keypoints on each face. Facial keypoints include points around the eyes, nose, and mouth on a face as shown in the image below



## <img src="images/pranay%20detected.png" alt="This is my face" width=400>

I have divided the workflow into 2 essential notebooks and a python file as follow:

* models.py : Define the Convolutional Neural Network architecture.
* Notebook 1 : Loading Data and Training the Convolutional Neural Network (CNN) to Predict Facial Keypoints
* Notebook 2 : Face and Facial Keypoint detection, Complete Pipeline.

## A General Outline:

Facial keypoints (also called facial landmarks) are the small magenta dots shown on each of the faces in the image above. In each training and test image, there is a single face and 68 keypoints, with coordinates (x, y), for that face. These keypoints mark important areas of the face: the eyes, corners of the mouth, the nose, etc. These keypoints are relevant for a variety of tasks, such as face filters, emotion recognition, pose recognition, and so on. 

In the below image, these keypoints are numbered, and you can see that specific ranges of points match different portions of the face.

<img src="images/keypoints%20numbered.jpg" width=350>

## Dataset

The dataset which has been used in this project has been extracted from the [YouTube Faces Dataset](https://www.cs.tau.ac.il/~wolf/ytfaces/), which includes videos of people in YouTube videos. These videos have been fed through some processing steps and turned into sets of image frames containing one face and the associated keypoints.

This facial keypoints dataset consists of 5770 color images. All of these images are separated into either a training or a test set of data.
* 3462 of these images are training images, to create a model to predict keypoints.
* 2308 are test images, to test the accuracy of your model.

The data for this project has been loaded from a separate workspace which is accessed via its URL. The information about the images and keypoints in this dataset are summarized in CSV files, which we can read in using pandas. For each data point(row) in CSV, it contains the image name, and 68 other columns describing the 38 distinguishing pairs of keypoints on the face. We can read the training CSV and get the annotations of 68 keypoints in an (N, 2) array where N is the number of keypoints(68) and 2 is the dimension of the keypoint coordinates (x, y).

## Training and testing the model

The CNN model architecture defined in the models.py file is used for training. After the model has been trained, you can feed the network any image that includes faces. The neural network expects a Tensor of a certain size as input. The inference of the keypoints in any test image typically happens over the course of the following steps:
* Detects all the faces in an image using a face detector (I have used a Haar Cascade detector for this project).
* Pre-process those face images so that they are grayscale, and transformed to a Tensor of the input size that my net expects.
* Use your trained model to detect facial keypoints on the image.


<b><p>This is an example of how my model performed on a test image consisting of me and my high school friends chilling at a cafe!<p></b>
<p><img src="images/my%20friends.jpg" alt="photo" width=600></p>
<b>Faces detected with keypoints marked on each of the detected face</b> <br>
<img src="images/rohit%20detected.png" alt="photo"> <img src="images/pranay%20detected.png" alt="photo"> <img src="images/shweta%20detected.png" alt="photo">
