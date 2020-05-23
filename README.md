### In this project I have build a facial keypoint detection system that takes in any image with faces, and predicts the location of 68 distinguishing keypoints on each face. Facial keypoints include points around the eyes, nose, and mouth on a face as shown in the image below

I have divided the workflow into 2 essential notebooks and a python file as follow:
* models.py : Define the Convolutional Neural Network architecture.
* Notebook 1 : Loading Data and Training the Convolutional Neural Network (CNN) to Predict Facial Keypoints
* Notebook 2 : Face and Facial Keypoint detection, Complete Pipeline.

## A General Outline:

Facial keypoints (also called facial landmarks) are the small magenta dots shown on each of the faces in the image above. In each training and test image, there is a single face and 68 keypoints, with coordinates (x, y), for that face. These keypoints mark important areas of the face: the eyes, corners of the mouth, the nose, etc. These keypoints are relevant for a variety of tasks, such as face filters, emotion recognition, pose recognition, and so on. Here they are, numbered, and you can see that specific ranges of points match different portions of the face.
