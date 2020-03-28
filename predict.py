import os
import sys
import cv2
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import matplotlib.patches as patches
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, scale_boxes
from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body

# Filter boxes having class scores > set threshold

def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = .6):
    box_scores = box_confidence * box_class_probs
    
    box_classes = K.argmax(box_scores, -1)
    box_class_scores = K.max(box_scores, -1)
    
    filtering_mask = box_class_scores > threshold

    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes = tf.boolean_mask(box_classes, filtering_mask)
    
    return scores, boxes, classes

# Non-max suppression to remove overlapping bounding boxes

def yolo_non_max_suppression(scores, boxes, classes, max_boxes = 10, iou_threshold = 0.5): 
    max_boxes_tensor = K.variable(max_boxes, dtype='int32')
    K.get_session().run(tf.variables_initializer([max_boxes_tensor]))
    
    nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes, iou_threshold)
    
    scores = K.gather(scores, nms_indices)
    boxes = K.gather(boxes, nms_indices)
    classes = K.gather(classes, nms_indices)
    
    return scores, boxes, classes

# Pipeline to filter boxes and nms

def yolo_eval(yolo_outputs, image_shape = (480., 640.), max_boxes=10, score_threshold=.6, iou_threshold=.5):
    
    box_xy, box_wh, box_confidence, box_class_probs = yolo_outputs
    boxes = yolo_boxes_to_corners(box_xy, box_wh)

    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = score_threshold)
    boxes = scale_boxes(boxes, image_shape)
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes, iou_threshold)
    
    
    return scores, boxes, classes

# Predict method which take the input image and outputs (scores, boxes, classes)

def predict(sess, image_file):
    image, image_data = preprocess_image('input/' + image_file, model_image_size = (608, 608))

    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],feed_dict = {yolo_model.input: image_data,K.learning_phase(): 0})

    print('Found {} boxes for {}'.format(len(out_boxes), image_file))
    colors = generate_colors(class_names)
    
    return out_scores, out_boxes, out_classes


if __name__ == '__main__':
	file_name = sys.argv[1]
	cv2.imwrite('input/{}'.format(file_name), cv2.resize(cv2.imread('input/{}'.format(file_name)), (640, 480)))

	sess = K.get_session()

	# Load yolo class names and anchors

	class_names = read_classes("model_data/coco_classes.txt")
	anchors = read_anchors("model_data/yolo_anchors.txt")
	image_shape = (480., 640.)

	# Loading pre-trained model

	yolo_model = load_model("model_data/yolo.h5")

	yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
	scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)
	out_scores, out_boxes, out_classes = predict(sess, file_name)

	classes = [class_names[out_class] for out_class in out_classes]
	color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

	fig, ax = plt.subplots(1)
	ax.set_axis_off()
	img = plt.imread('input/{}'.format(file_name))
	for i in range(len(out_classes)):
	    s = (out_boxes[i][1], out_boxes[i][0])
	    e = (out_boxes[i][3], out_boxes[i][2])
	    rect = patches.Rectangle(s, abs(s[0] - e[0]), abs(s[1] - e[1]), edgecolor = color[i % len(color)], facecolor = 'none')
	    plt.text(s[0] + 10, s[1] + 25, classes[i], fontsize = 10, weight = 'heavy', color = 'black', bbox=dict(facecolor = color[i % len(color)], alpha = 1))
	    ax.imshow(img)
	    ax.add_patch(rect)
	plt.show()