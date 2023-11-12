# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 08:11:10 2020

@author: Spyridon Montesantos

Basic routine library for dev.ipynb and app.py.
It contains all the subroutines used to perform face and landmark detection and imprint the results on the input video frames.

"""

#%%
import numpy as np
import cv2,dlib
import os
import matplotlib.pyplot as plt



#%% 
################################################################
# Transform dlib.rect to normal cv2 bounding box
# Input:    rect        --> Front face detector object, output from dlib.get_frontal_face_detector(image,1)
# Output:   (x,y,w,h)   --> The edge coordinates, width and height of the detection rectangle. 
def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
 
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)


#%%
################################################################
# Transform dlib.predictor landmark object to coordinates array.
# Input:    shape --> Shape predictor, output from dlib.shape_predictor(image, dlib.rect object)
#           nb_landmarks --> number of landmarks expected from the predictor.   
def shape_to_np(shape, nb_landmarks = 68, dtype="int"):
    # nb_landmarks=68
    
	# initialize the list of (x, y)-coordinates
    coords=np.zeros((nb_landmarks,2), dtype=dtype)
 
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
    for i in range(0,nb_landmarks):
        coords[i]=(shape.part(i).x,shape.part(i).y)
    
	# return the list of (x, y)-coordinates
    return coords

#%%
################################################################
# Maintain the largest detected rectangle as the detected front face in the image.
# Input:    rects --> Front face detector object, output from dlib.get_frontal_face_detector(image,1)
# Output:   only the single larges dlib.rect object contained in the input (AREA)
def largest_front(rects):
    a0=0
    idx=0
    for (i,dlib.rect) in enumerate(rects):
        # Call subfunction rect_to_bb to obtain box coordinates and compare sizes
        (x,y,w,h)=rect_to_bb(dlib.rect)
        if w*h>a0:
            a0=w*h
            idx=i
            
    # Maintain the largest detected rectangle for measurements
    dlib.rect=list(enumerate(rects))[idx][1]
    return dlib.rect




#%%
################################################################
# Get the left and right eye bboxes and their aspect ratios.
# Input:    landmarks --> Front face predictor object, predictor(img, face_bbox) after processing with shape_to_np subroutine.
# Output:   2 lists left_eye_bbox and right_eye_bbox with the coordinates of the left, top, right and bottom pixels of the box.
def eye_bboxes(landmarks):
    left_eye = landmarks[36:42]
    right_eye = landmarks[42:48]

    # Left eye bbox.
    left_eye_bbox = [left_eye.min(axis=0)[0], left_eye.min(axis=0)[1],
                 left_eye.max(axis=0)[0], left_eye.max(axis=0)[1]]

    # Right eye bbox.
    right_eye_bbox = [right_eye.min(axis=0)[0], right_eye.min(axis=0)[1],
                 right_eye.max(axis=0)[0], right_eye.max(axis=0)[1]]
    return left_eye_bbox, right_eye_bbox


#%%
################################################################
# Imprint the face bbox, the detected landmarks and the two eye bboxes on the image.
# Input:    img -->     the image on which we want to imprint the detected features.
#           principal_face -->  output of face detector and the largest_front subroutine.
#           landmarks -->   Front face predictor object, predictor(img, face_bbox) after processing with shape_to_np subroutine.
#           left_eye_bbox, right_eye_bbox -->   The eye bounding boxes, outputs of the eye_bboxes subroutine.
# Output:   The input image with the featuresi mprinted on it.
def imprint_on_img(img, principal_face, landmarks, left_eye_bbox, right_eye_bbox):
     # Print face bbox, landmarks and the eye bboxes on the image for verification.
    cv2.rectangle(img, (principal_face.left(), principal_face.top()),
                (principal_face.right(), principal_face.bottom()), (255, 0, 0), 12);

    for (x, y) in landmarks:
        cv2.circle(img, (x, y), 8, (255, 0, 0), -1);

    cv2.rectangle(img, (left_eye_bbox[0], left_eye_bbox[1]),
                (left_eye_bbox[2], left_eye_bbox[3]), (0, 255, 0), 6);

    cv2.rectangle(img, (right_eye_bbox[0], right_eye_bbox[1]),
                (right_eye_bbox[2], right_eye_bbox[3]), (0, 0, 255), 6);

    return img


#%%
################################################################
# Determine if sleepy condition.
# Input:    left_eye_bbox, right_ eye_bbox --> The outputs of the eye_bboxes subroutine, 
#               2 lists left_eye_bbox and right_eye_bbox with the coordinates of the left, 
#               top, right and bottom pixels of the box.
#           thresh  -->     The bbox aspect ration threshold where the eyes are considered closed; here set at 0.15.
# Output:   sleep   --> A boolean, True if sleepy or False if not.
# NOTE:     This function will be improved at later stages. Instead of bboxes, we can use the distance between 
#           equivalent top-bottom landmarks or, even better, a classifier on the entire face.
def sleepy(left_eye_bbox, right_eye_bbox, thresh = 0.15):
    sleep = False

    # Get the bbox aspect ratios: (top-bottom) / (right-left)
    left_eye_bbox_ar = (left_eye_bbox[3] - left_eye_bbox[1]) / (left_eye_bbox[2] - left_eye_bbox[0])
    right_eye_bbox_ar = (right_eye_bbox[3] - right_eye_bbox[1]) / (right_eye_bbox[2] - right_eye_bbox[0])

    # We will choose the threshold for sleep to be at 0.15 aspect ratio.
    if (left_eye_bbox_ar <= thresh) & (right_eye_bbox_ar <= thresh):
        sleep = True

    return sleep