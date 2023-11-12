# Import libraries
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

def main():
    st.set_page_config(layout="wide")
    st.title("Sleep Detection Web App", )

    uploaded_file = st.file_uploader("Upload a video file", type=["avi", "mp4"])

    if uploaded_file is not None:
        
        # Display the original video
        
        st.header("Original Video")
        # st.video(uploaded_file)

        DEFAULT_WIDTH = 25

        width = st.sidebar.slider(
            label="Video size", min_value=0, max_value=100, value=DEFAULT_WIDTH, format="%d%%"
        )

        width = max(width, 0.01)
        side = max((100 - width) / 2, 0.01)

        _, container, _ = st.columns([side, width, side])
        container.video(data=uploaded_file)


        


    
if __name__ == "__main__":
    # Run the Streamlit app
    main()