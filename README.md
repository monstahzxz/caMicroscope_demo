# Object Detection using YOLO

## Code Challenge
Using a machine learning toolkit of your choice, create a tool which identifies objects in the image, then returns positions in pixels corresponding to bounding boxes of a user-selected class of object in the image. For example, given an image with both cats and dogs, return bounding boxes for only cats.

## Solution
I have set up a solution which takes in an input image and outputs <b>bounding boxes</b> of regions of interest. As suggested in my proposal, I have set up an object detector using <b>Yolo</b> to specifically extract patches corresponding to objects of interest. The Yolo model is able to detect about 80 classes in total. This solution is just to demonstrate the ability to easily crop out <b>regions of interest</b>. See sample input/output examples given below.

## Architecture
* Yolov2 (DarkNet)

## Tools and References Used
* Tensorflow (Keras)
* Matplotlib
* [Yolov2 Pre Trained Model (DarkNet)](https://pjreddie.com/darknet/yolo/)
* [OverFeat Paper](https://arxiv.org/abs/1312.6229)

## Example (Input/Output)
* <b>Input</b><br>
<img src="https://github.com/monstahzxz/caMicroscope_demo/examples/input.jpeg"/>
* <b>Output</b><br>
<img src="https://github.com/monstahzxz/caMicroscope_demo/examples/output.jpeg"/>
