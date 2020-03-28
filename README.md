# Object Detection using YOLO

## Code Challenge
Using a machine learning toolkit of your choice, create a tool which identifies objects in the image, then returns positions in pixels corresponding to bounding boxes of a user-selected class of object in the image. For example, given an image with both cats and dogs, return bounding boxes for only cats.

## Solution
I have set up a solution which takes in an input image and outputs <b>bounding boxes</b> of regions of interest. As suggested in my proposal, I have set up an object detector using <b>Yolo</b> to specifically extract patches corresponding to objects of interest. The Yolo model is able to detect about 80 classes in total. This solution is just to demonstrate the ability to easily crop out <b>regions of interest</b>. See sample input/output examples given below.

## Steps to Setup Locally
1. Clone the repository
2. Extract model_data.rar to directory model_data
3. Place your image in images directory
4. Run the following command to see results
``` sh
  py predict.py filename_in_images_directory
```

## Architecture
* Yolov2 (DarkNet)

## Tools and References Used
* Tensorflow (Keras)
* Matplotlib
* [Yolov2 Pre Trained Model (DarkNet)](https://pjreddie.com/darknet/yolo/)
* [OverFeat Paper](https://arxiv.org/abs/1312.6229)

## Example (Input/Output)
* <b>Input</b><br>
<img height="200" width="300" src="https://github.com/monstahzxz/caMicroscope_demo/blob/master/examples/input.jpeg"/>
* <b>Output</b><br>
<img height="200" width="300" src="https://github.com/monstahzxz/caMicroscope_demo/blob/master/examples/output.jpeg"/>

## Video Example
The model was run on a video to do real-time object detection. Please follow the links below to see the results.
* [Input](https://drive.google.com/open?id=1JrLAiMroWTkXitvCLb26ZvKzgezGacTR)
* [Output](https://drive.google.com/open?id=1-TTYYSvtZYYmcLEg7oGb-8pBsN7BiPXE)
