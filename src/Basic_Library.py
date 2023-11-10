# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 08:11:10 2020

@author: Spyridon Montesantos

Basic routine library for Basic_ProfileFace_Detection_Script.py and Basic_FrontFace_Detection_Script.

"""

#%%
import cv2,dlib
import numpy as np
from scipy.spatial import distance
from tkinter.filedialog import askopenfilename
import os
import matplotlib.pyplot as plt
import copy


#%%
################################################################
# Reshape and pad image to predetermined width,height. The padding is noise.
# Inputs:   image           --> the image for resizing and padding
#           nWidth, nHeight --> the new width and height for the padded image
# Outputs:  repad_info      --> a dictionary containing the repaded image, and the offset of the input image to center to the paded image.
def resize_pad(image,w,h, repad_choice):
    
    
    height,width,cc=image.shape
    
    
    if repad_choice==1:
        
        # repad_choice=1 --> Resize input so that the maximum dimension of the output is a scaled version of the maximum dimension of the input. The minimum dimension is determined using the same scaling factor to maintain the aspect ratio. 
        if height>=width:
            nHeight=max([h,w])
            nWidth=min([h,w])
            r=nHeight/height
        
        else:
            nHeight=min([h,w])
            nWidth=max([h,w])
            r=nWidth/width
    
        nh=np.uint16(np.round(height*r))
        nw=np.uint16(np.round(width*r))
        repaded=cv2.resize(image,(nw, nh), interpolation=cv2.INTER_AREA)
        xx=repaded.shape[1]//2
        yy=repaded.shape[0]//2
        
        print('Input image resized to h=' + str(nh) + ' , w=' + str(nw))
        
    elif repad_choice==2:
        
        # Resize input image with respect to maximum output dimensions (aspect ratio of the output image remains pre-determined.
        if height>=width:
            nHeight=max([h,w])
            nWidth=min([h,w])
            r=nWidth/width
        
        else:
            nHeight=min([h,w])
            nWidth=max([h,w])
            r=nHeight/height
    
        nh=np.uint16(np.round(height*r))
        nw=np.uint16(np.round(width*r))
        resized=cv2.resize(image,(nw, nh), interpolation=cv2.INTER_AREA)
        
    
        # Create a new image for padding with the expected output shape.
        repaded=np.random.randint(0, 256, (nHeight,nWidth,cc), dtype=np.uint8)
    
        # Blur the repaded image
        repaded=cv2.blur(repaded,(15,15))
    
        # Get the offset center; it will be zero along the max dimension
        xx=(repaded.shape[1]-resized.shape[1])//2
        yy=(repaded.shape[0]-resized.shape[0])//2
    
    
        # Integrate input resized input image to the repaded noisy image using the
        # calculated offset center.
        repaded[yy:yy+resized.shape[0], xx:xx+resized.shape[1]]=resized
        
        print('Input image resized and padded to h=' + str(nHeight) + ' , w=' + str(nWidth))
        
    else:
        repaded=image
        xx=image.shape[1]//2
        yy=image.shape[0]//2
        print('Input image not changed')
    
    
    # Create dictionary output
    repad_info={'repaded_image': repaded,
                'dHeight': yy*2,
                'dWidth': xx*2}
    
    # cv2.imwrite({get path}, repaded)
    return repad_info



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



################################################################
# Maintain the largest detected rectangle as the detected profile face in the image.
# Input:    rects --> Front face Haarcascade rectangle - (n,4) array (x,y,w,h)
# Output:   only the single largest rectangle contained in the input (AREA)
def largest_profile(rects):
    a0=0
    idx=0
    for i in range(rects.shape[0]):
        x=rects[i,0]
        y=rects[i,1]
        w=rects[i,2]
        h=rects[i,3]
        
        if w*h>a0:
            a0=w*h
            idx=i
            
    # Maintain the largest detected rectangle for measurements
    rect=rects[idx,:]
    return rect


#%%
################################################################
# Obtain landmark pixel measurements the front faces detected in the image. Works only on single set of landmarks.
# Input: landmark_front --> Needs the results from the  the dlib.shape_predictor(img, dlib.rect) transformed into np array (subfunction shape_to_np). 
def landmark_measurements_front(landmarks_front):
    # # List front landmarks positionally
    # Jawline=list(range(0,17))
    # Right_Eyebrow=list(range(17,22))
    # Left_Eyebrow=list(range(22,27))
    # Nose=list(range(27,36))
    # Right_Eye=list(range(36,42))
    # Left_Eye=list(range(42,48))
    # Mouth_Out=list(range(48,61))
    
    # Init output.
    dist_array=np.zeros(6)
    dist_front_px_dict={}
    
    
    # Nose width distance in pixels.
    p1=landmarks_front[31,:]    # Right nostril edge
    p2=landmarks_front[35,:]    # Left nostril edge
    d=distance.euclidean(p1,p2)
    dist_front_px_dict.update({'NoseWidth': round(d,3)})
    dist_array[0]=round(d,3)
    
    
    # Nose height measured from the nose bridge indentation.
    p1=landmarks_front[27,:]    # Nose bridge indentation.
    p2=landmarks_front[33,:]    # Nose base.
    d=distance.euclidean(p1,p2)
    dist_front_px_dict.update({'NoseHeight_bridge': round(d,3)})
    dist_array[1]=round(d,3)
    
    
    # Nose height measured from the eye level. Use the average of the two inner eye edges.
    p1=landmarks_front[39,:]    # Right eye inner edge.
    p2=landmarks_front[42,:]    # Left eye inner edge.
    p3=landmarks_front[33,:]    # Nose base
    d=distance.euclidean((p1+p2)/2, p3)
    dist_front_px_dict.update({'NoseHeight_eyelvl': round(d,3)})
    dist_array[2]=round(d,3)
    
    
    # Distance between nose bridge indentation and chin crease - corrected for open mouth [1/3rd of distance between lower lip and chin].
    p1=landmarks_front[27,:]    # Nose bridge indentation.
    p2=landmarks_front[57,:]    # Lower lip.
    p3=landmarks_front[8,:]     # Chin.
    d=distance.euclidean(p1,p2) + distance.euclidean(p2,p3)/3
    dist_front_px_dict.update({'Bridge-Crease_bridge': round(d,3)})
    dist_array[3]=round(d,3)
    
    
    # Distance between nose @ eyel level and chin crease - corrected for open mouth [1/3rd of distance between lower lip and chin]
    p1=landmarks_front[39,:]    # Right eye inner edge.
    p2=landmarks_front[42,:]    # Left eye inner edge.
    p3=landmarks_front[57,:]    # Lower lip.
    p4=landmarks_front[8,:]     # Chin.
    d=distance.euclidean((p1+p2)/2,p3) + distance.euclidean(p3,p4)/3
    dist_front_px_dict.update({'Bridge-Crease_eyelvl': round(d,3)})
    dist_array[4]=round(d,3)
    
    
    # Vertical distance between outer eye edge and lower lip. The distance of the left eye outer edge to the line defined by the lip edges is calculated.
    p1=landmarks_front[48,:]    # Mouth right edge.
    p2=landmarks_front[54,:]    # Mouth left edge.
    p3=landmarks_front[45,:]    # Left eye outer edge.
    d=abs(np.cross(p2-p1,p3-p1)/np.linalg.norm(p2-p1))
    dist_front_px_dict.update({'Eye_Llip_vertical': round(d,3)})
    dist_array[5]=round(d,3)
    
    dist_front_px_dict.update({'Distance_array': dist_array})
    
    
    return dist_front_px_dict
    

################################################################
# Detection visualization - front. Visualize the results of object detection and shape prediction with the processed image.
# NOTE: Future implementation to use adaptable font sizes.
def detection_visualization_dlib(image, rect, landmarks, imagePath):
    # Call subfunction rect_to_bb to obtain box coordinates from input rect.
    (x,y,w,h)=rect_to_bb(rect)
    
    # Display box in the image.
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 8)
    
    # Show the face number
    cv2.putText(image,"face#{}".format(1),(x-10,y-10),
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=2,color=(255,255,0),lineType=20)
    
    # Display the face landmarks on the image
    for i in range(len(landmarks)):
        xc=landmarks[i,0]
        yc=landmarks[i,1]
        cv2.circle(image,(xc,yc),1,(0,0,255),6)
    
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    plt.imshow(image)

    path="C:/Users/smont/Desktop/New folder/"
    spl_char1='/'
    spl_char2='.'
    b=imagePath[imagePath.rfind(spl_char1)+1:imagePath.rfind(spl_char2)]
    plt.imsave(path+b+'_out.png',image)
    
    print('Detected face and landmarks stored in ' + path+b+'_out.png')
    return(image)



################################################################
# Transform measurements to real world dimensions and output to .txt file.
# Input: front_distances_px     -->  Uses the results from the landmark_measurements_front subfunction. 
#       scale obj_px, scale_obj_cm --> The same dimension of the scaling object in pixels and cm.
#       path    --> The path where the .txt file will be stored.
def output_front_measurements(front_distances_px, scale_obj_px, scale_obj_cm, predictor_path, img_path):
    
    # Open measurement text file where measurements are recorded.
    front_measurements=open(predictor_path+"front_measurements.txt",'w')
    front_measurements.write("File containing the measurements conducted on the detected facial fronts. Image path: \n" + img_path +"\n")
    front_measurements.write("The format is: Nb of measurement, Description of measurement, <TAB> Measurement in pixels <TAB> Measurement in cm \n\n")

    # Use the front_distances_px dictionary and the scale_object_front_px and scale_object_front_cm to transform measurements to cm and export to document. Call subfunction output_measurements.
    front_distances_cm=copy.deepcopy(front_distances_px)
    
    
    d_px=front_distances_px['Distance_array']
    d_cm=d_px*scale_obj_cm[0]/scale_obj_px[0]
    
    
    front_distances_cm['Distance_array']=d_cm
    
    # NOSE WIDTH DIMENSIONS: CORRECTION BY 1cm FOR INNER NOSTRIL LANDMARKS AND COMFORT!
    d_cm[0]=d_cm[0]+1
    front_distances_cm['NoseWidth']=d_cm[0]
    front_measurements.write('1. Nose width \t' +str(round(d_px[0],3))+'\t'+str(d_cm[0])+'\n')
    
    front_distances_cm['NoseHeight_bridge']=d_cm[1]
    front_measurements.write('2. Nose height measured at the bridge indentation \t' +str(d_px[1])+'\t'+str(round(d_cm[1],3))+'\n')
    
    front_distances_cm['NoseHeight_eyelvl']=d_cm[2]
    front_measurements.write('3. Nose height measured at inner eye level \t' +str(d_px[2])+'\t'+str(round(d_cm[2],3))+'\n')
    
    front_distances_cm['Bridge-Crease_bridge']=d_cm[3]
    front_measurements.write('4. Nose bridge - chin crease distance \t' +str(d_px[3])+'\t'+str(round(d_cm[3],3))+'\n')
    
    front_distances_cm['Bridge-Crease_eyelvl']=d_cm[4]
    front_measurements.write('5. Inner eye level - chin crease distance \t' +str(d_px[4])+'\t'+str(round(d_cm[4],3))+'\n')
    
    front_distances_cm['Eye_Llip_vertical']=d_cm[5]
    front_measurements.write('6. Outer eye - lower lip vertical distance \t' +str(d_px[5])+'\t'+str(round(d_cm[5],3))+'\n')
      
    # Close .txt file.
    front_measurements.close()
    
    return(front_distances_cm)


#%% 
################################################################
# Calculate point p3 projection on a line defined by points p1 and p2.
# Input:    p1,p2,p3 --> coordinates of the 3 points.
# Output:   p4 --> coordinates of the projection.
def proj_a_on_b(p1,p2,p3):
    a=p3-p1
    b=p2-p1
    
    a=np.asarray(a)[0]
    b=np.asarray(b)[0]
    
    costheta=np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
    
    p4=p1+np.linalg.norm(a)*costheta*(b/np.linalg.norm(b))
    p4=np.round(p4,0)
    return p4
    
    
    

################################################################
# Obtain landmark pixel measurements the front faces detected in the image. Works only on single set of landmarks.
# Input: landmark_front --> Needs the results from the the dlib.shape_predictor(img, dlib.rect) transformed into np array (subfunction shape_to_np). 
def landmark_measurements_profile(landmarks_profile):
        
    # Init output.
    dist_array=np.zeros(7)
    dist_profile_px_dict={}
    
    # Nose height measured from the nose bridge indentation.
    p1=landmarks_profile[18,:]  # Nose bridge indentation.
    p2=landmarks_profile[23,:]  # Nose base.
    d=distance.euclidean(p1,p2)
    dist_profile_px_dict.update({'NoseHeight_bridge': round(d,3)})
    dist_array[0]=round(d,3)
    
        
    # Nose height measured from the eye level. Calculated by projecting eye outer edge point on line determined by nose base and nose bridge indentation.
    p1=landmarks_profile[23,:]  # Nose base.
    p2=landmarks_profile[18,:]  # Nose bridge indentation.
    p3=landmarks_profile[12,:]  # Eye outer edge.
    
    p4=proj_a_on_b(p1,p2,p3)
    d=distance.euclidean(p1,p4)
    dist_profile_px_dict.update({'NoseHeight_eyelvl': round(d,3)})
    dist_array[1]=round(d,3)
    
    
    # Distance between nose bridge indentation and chin crease - corrected for open mouth[1/3rd of distance between lower lip and chin].
    p1=landmarks_profile[18,:]  # Nose bridge indentation.
    p2=landmarks_profile[28,:]  # Lower lip.
    p3=landmarks_profile[8,:]   # Chin
    d=distance.euclidean(p1,p2) + distance.euclidean(p2,p3)/3
    dist_profile_px_dict.update({'Bridge-Crease_bridge': round(d,3)})
    dist_array[2]=round(d,3)
    
    
    # Distance between nose @ eyel level and chin crease - corrected for open mouth [1/3rd of distance between lower lip and chin]. Calculated by projectin eye outer edge point on line determined by nose base and nose bridge indentation.
    p1=landmarks_profile[23,:]  # Nose base.
    p2=landmarks_profile[18,:]  # Nose bridge indentation.
    p3=landmarks_profile[12,:]  # Eye outer edge.
    
    p4=proj_a_on_b(p1,p2,p3)
    
    p5=landmarks_profile[28,:]  # Lower lip.
    p6=landmarks_profile[8,:]   # Chin.
    
    d=distance.euclidean(p4,p5) + distance.euclidean(p5,p6)/3
    dist_profile_px_dict.update({'Bridge-Crease_eyelvl': round(d,3)})
    dist_array[3]=round(d,3)
    
    
    # Nose length - base to nose tip, diagonal.
    p1=landmarks_profile[23,:]  # Nose base.
    p2=landmarks_profile[21,:]  # Nose tip.
    d=distance.euclidean(p1,p2)
    dist_profile_px_dict.update({'NoseLength_diagonal': round(d,3)})
    dist_array[4]=round(d,3)
    
    
    # Nose length - base to nose tip, straight from face plane.
    p1=landmarks_profile[23,:]  # Nose base.
    p2=landmarks_profile[18,:]  # Nose bridge indentation.
    p3=landmarks_profile[21,:]  # Nose tip
    
    p4=proj_a_on_b(p1,p2,p3)
    d=np.linalg.norm(p4-p3)
    dist_profile_px_dict.update({'NoseLength_straight': round(d,3)})
    dist_array[5]=round(d,3)
    
    
    # Vertical distance between outer eye edge and lower lip. Calculated by projecting eye outer edge on the mouth line.
    p1=landmarks_profile[27,:]  # Upper inner lip.
    p2=landmarks_profile[29,:]  # Lower inner lip.
    p3=(p1+p2)/2
    
    p4=landmarks_profile[25,:]  # Mouth edge.
    p5=landmarks_profile[12,:]  # Eye outer edge.
    
    p6=proj_a_on_b(p3, p4, p5)
    d=np.linalg.norm(p6-p5)
    dist_profile_px_dict.update({'Eye_Llip_vertical': round(d,3)})
    dist_array[6]=round(d,3)
    
    
    dist_profile_px_dict.update({'Distance_array': dist_array})
    
    
    return dist_profile_px_dict


################################################################
# Detection visualization - profile. Visualize the results of object detection and shape prediction with the processed image.
# NOTE: Future implementation to use adaptable font sizes.
def detection_visualization_haar(image, rect, landmarks, imagePath):
    
    (x,y,w,h)=rect
    
    # Display box in the image.
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 3)
    
    # Show the face number
    cv2.putText(image,"face#{}".format(1),(x-10,y-10),
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=2,color=(255,255,0),lineType=8)
    
    # Display the face landmarks on the image
    for i in range(len(landmarks)):
        xc=landmarks[i,0]
        yc=landmarks[i,1]
        cv2.circle(image,(xc,yc),1,(0,0,255),3)
        # cv2.putText(image, str(i), (xc,yc), fontFace=cv2.FONT_HERSHEY_PLAIN,
        #             fontScale=1, color=[0,255,0])
    
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    plt.imshow(image)

    path="C:/Users/smont/Desktop/New folder/"
    spl_char1='/'
    spl_char2='.'
    b=imagePath[imagePath.rfind(spl_char1)+1:imagePath.rfind(spl_char2)]
    plt.imsave(path+b+'_out.png',image)
    
    print('Detected face and landmarks stored in ' + path+b+'_out.png')
    return(image)


################################################################
# Transform measurements to real world dimensions and output to .txt file.
# Input: front_distances_px     -->  Uses the results from the landmark_measurements_front subfunction. 
#       scale obj_px, scale_obj_cm --> The same dimension of the scaling object in pixels and cm.
#       path    --> The path where the .txt file will be stored.
def output_profile_measurements(profile_distances_px, scale_obj_px, scale_obj_cm, predictor_path, img_path):
    
    # Open measurement text file where measurements are recorded.
    profile_measurements=open(predictor_path+"profile_measurements.txt",'w')
    profile_measurements.write("File containing the measurements conducted on the detected facial profile. Image path: \n" + img_path +"\n")
    profile_measurements.write("The format is: Nb of measurement, Description of measurement, <TAB> Measurement in pixels <TAB> Measurement in cm \n\n")

    # Use the front_distances_px dictionary and the scale_object_front_px and scale_object_front_cm to transform measurements to cm and export to document. Call subfunction output_measurements.
    profile_distances_cm=copy.deepcopy(profile_distances_px)
    
    d_px=profile_distances_px['Distance_array']
    d_cm=d_px*scale_obj_cm[0]/scale_obj_px[0]
    
    profile_distances_cm['Distance_array']=d_cm

    
    profile_distances_cm['NoseHeight_bridge']=d_cm[0]
    profile_measurements.write('1. Nose height measured at the bridge indentation \t' +str(d_px[0])+'\t'+str(round(d_cm[0],3))+'\n')
    
    profile_distances_cm['NoseHeight_eyelvl']=d_cm[1]
    profile_measurements.write('2. Nose height measured at eye level \t' +str(d_px[1])+'\t'+str(round(d_cm[1],3))+'\n')
    
    profile_distances_cm['Bridge-Crease_bridge']=d_cm[2]
    profile_measurements.write('3. Nose bridge - chin crease distance \t' +str(d_px[2])+'\t'+str(round(d_cm[2],3))+'\n')
    
    profile_distances_cm['Bridge-Crease_eyelvl']=d_cm[3]
    profile_measurements.write('4. Eye level - chin crease distance \t' +str(d_px[3])+'\t'+str(round(d_cm[3],3))+'\n')
    
    profile_distances_cm['NoseLength_diagonal']=d_cm[4]
    profile_measurements.write('5. Nose length - diagonal \t' +str(d_px[4])+'\t'+str(round(d_cm[4],3))+'\n')
    
    profile_distances_cm['NoseLength_straight']=d_cm[5]
    profile_measurements.write('6. Nose length - straight \t' +str(d_px[5])+'\t'+str(round(d_cm[5],3))+'\n')
    
    profile_distances_cm['Eye_Llip_vertical']=d_cm[6]
    profile_measurements.write('7. Outer eye - lower lip vertical distance \t' +str(d_px[6])+'\t'+str(round(d_cm[6],3))+'\n')
      
    # Close .txt file.
    profile_measurements.close()
    
    return(profile_distances_cm)