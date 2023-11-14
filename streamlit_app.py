# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import base64
import io
import tempfile

# Image processing libraries
import cv2 
import dlib
import imageio
import streamlit as st

# Local library
import Basic_Library as blib



def main():
    st.set_page_config(layout="wide")
    st.title("Sleep Detection Web App", )

    explainer_text = '''This is a webapp simulating the activity of a sleep detector. The user is asked to import a video containing faces. The video is uploaded and processed and the faces and eyes are detected. If the eyes of the principal face in the image (i.e. the face with the largest bounding box) close there is a visual alarm. In later development there will also be a sound alarm.'''

    st.markdown(explainer_text)
    uploaded_file = st.file_uploader("Upload the video file - preferably w:480 - h:640", type=["avi", "mp4"])

    if uploaded_file is not None:

        # Need to store the uploaded vide to temporary file in order to be able to use the cv2 VideoCapture library.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(uploaded_file.read())

        # Perform video processing.
        # Select path and import video.
        cap = cv2.VideoCapture(temp_file.name)

        # Get the video properties.
        fps = cap.get(cv2.CAP_PROP_FPS)
        # target_fps = fps / 6
        fwidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fheight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Import face detector.
        detector = dlib.get_frontal_face_detector()

        # Import landmark predictor.
        landmark_predictor_path = os.path.join('models', 'shape_predictor_68_face_landmarks.dat' )
        predictor = dlib.shape_predictor(landmark_predictor_path)


        # Loop through the uploaded video frames, make modifications and collate into new video.
        modified_frames = []
        count = 0
        st.subheader('Processing:')
        progress_bar = st.progress(0)

        while True:
            ret, frame = cap.read()
    
            if not ret:
                break
    
            # Call function frame_processing to obtain the modified frame and whether in the frame
            # the eyes are detected to be closed.
            modified_frame, eyes_closed = blib.frame_processing(frame, detector, predictor)

            # Append the modified frame to the list
            modified_frames.append(modified_frame)

            # Track algorithm progress.
            # print('Frame processed: ', count)
            # count += 1

            # Track algorithm progress.
            progress_percentage = count / total_frames 
            progress_bar.progress(progress_percentage)
            count += 1

        cap.release()

        # Create the processed video using the modified frames and store to memory. An option will be given
        # to store in the hard drive by the user.
        processed_video = blib.video_creator_264_to_memory(modified_frames, fps)


        # Display the processed video.
        DEFAULT_WIDTH = 40

        width = st.sidebar.slider(
            label="Video size", min_value=0, max_value=100, value=DEFAULT_WIDTH, format="%d%%"
        )

        width = max(width, 0.01)
        side = max((100 - width) / 2, 0.01)

        _, container, _ = st.columns([side, width, side])
        container.video(data=processed_video)



if __name__ == "__main__":
    # Run the Streamlit app
    main()