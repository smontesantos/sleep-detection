# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 10:26:03 2020

@author: Spyridon.MONTESANTOS

Script designed to detect and display the human face and its landmarks 
frontally. 
For the frontal face and its landmarks, the dlib face detector is used (SVM
+ HOG)
                                                                        
    1. Import image and pre-trained 68 point facial landmark predictor
    2. Detect frontal face location in an image using the
    dlib.get_frontal_face_detector() tool
    3. Detect facial landmarks using the pretrained 68 point facial 
    landmark predictor
    4. For each detected face, display bounding box and facial landmarks

"""

#%%
import cv2,dlib
import numpy as np
from scipy.spatial import distance
# from tkinter.filedialog import askopenfilename
import os
import matplotlib.pyplot as plt
import copy
import Basic_Library as blib
from tkinter import filedialog
from tkinter import *
  

########################################################################
########################### MAIN SCRIPT ################################
#%% INITIALIZE VARIABLES
predictor_path=os.getcwd()+'\\DetPred\\'

# Option to resize and pad the image.
# repad_choice=0 --> use the input image
# repad_choice=1 --> resize image with a scaling ratio between its max dimension and the defined ndim max dimension.
# repad_choice=2 --> resize and pad the image to dimensions defined by ndim. The padding is white noise.
repad_choice=1
ndim_front=(1280,720) # Y x X, or Height x Width --> This will represent max and min dimension size.

# NOTE: Replace with object dimensions per image. In future implementations, automatic object detection will be performed.
scale_object_front_px=(227,158)   # width x height
scale_object_front_cm=(8.5, 5.4) # Credit card dimensions 

scale_object_profile_px=(250,158)   # width x height
scale_object_profile_cm=(8.56, 5.4) # Credit card dimensions  --> The distance from the Eye to lower lip can also be used as it the landmarks are common between front and profile.




#%% FILE and PATH SELECTION - FACIAL FRONT

# Dialog box to select image
root=Tk()
root.imagePath_front=filedialog.askopenfilename(filetypes= (("All files","*.*"),("Image files","*amphas*.jpg")))
root.update()
imagePath_front=root.imagePath_front
root.destroy()
# cascPath=askopenfilename(filetypes= (("Haarcascade files","haar*.xml"),("Allfiles","*.*")))

# Search the current directory for the landmark shape predictor
filenames=os.listdir(predictor_path)
for idx in range(len(filenames)):
    if 'shape_predictor_68_face_landmarks.dat' in filenames[idx]:
        shapePath_front=os.path.join(predictor_path,filenames[idx])
        print('Shape Predictor: '+shapePath_front)

del filenames,idx

# Import predictor (pre-trained, from file), detector (from dlib) and image
predictor_front=dlib.shape_predictor(shapePath_front)
detector_front=dlib.get_frontal_face_detector()
print('Frontal face detector integrated in dlib library')
image_front=cv2.imread(imagePath_front)
del shapePath_front


#%% OPTIONAL - RESIZE AND PAD IMAGE
# Use the option repad_choice: 0--> no change, 1--> resize, 2--> resize and pad,

repaded_front=blib.resize_pad(image_front,ndim_front[1], ndim_front[0], repad_choice)['repaded_image']



#%% DETECT ALL FACES - FACIAL FRONTS
# Transform rgb image to grayscale
gray_front=cv2.cvtColor(repaded_front, cv2.COLOR_BGR2GRAY)

# Detect the faces in the grayscale image using dlib detector.
rects=detector_front(gray_front,1)
nb_faces_front=len(list(enumerate(rects)))
print("Front facial detector found {0} faces".format(nb_faces_front))


#%% CONDUCT MEASUREMENTS - LARGEST DETECTED FACIAL FRONT
front_distances_px=[]
if nb_faces_front==0:
    # If no faces are found, state in text file.
    print('FACES NOT FOUND - No measurements can be conducted.')

else:
    # For each face, get containing rectangle and check which is the largest. The measurements will be conducted on the largest detected face. Call subfunction largest_front
    dlib.rect=blib.largest_front(rects)
    
    
    # Detect the landmarks on frontal face using dlib pretrained predictor
    landmarks_front=predictor_front(gray_front,dlib.rect)
    # Call subfunction shape_to_np to transform landmarks to numpy array.
    landmarks_front=blib.shape_to_np(landmarks_front, nb_landmarks=68)
    
        
    # Call subfunction landmark_measurements_front to obtain the measurements relevant to mask sizing (in pixels).
    front_distances_px=blib.landmark_measurements_front(landmarks_front)
    
    # Visualize output window and landmarks. Call subfunction visualize_detection
    repaded_front=blib.detection_visualization_dlib(repaded_front, dlib.rect, landmarks_front, imagePath_front)


#%% OUTPUT FRONTAL MEASUREMENTS
# Use the front_distances_px dictionary and the scale_object_front_px and scale_object_front_cm to transform measurements to cm and export to document. Call subfunction output_measurements.
if len(front_distances_px)>0:
    front_distances_cm=blib.output_front_measurements(front_distances_px, scale_object_front_px, scale_object_front_cm, predictor_path, imagePath_front)

# del gray_front, image_front




    