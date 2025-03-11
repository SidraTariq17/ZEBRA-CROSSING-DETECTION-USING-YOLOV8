# import streamlit as st
# import cv2
# import numpy as np
# from ultralytics import YOLO
#
#
# # Load the YOLO model
# model = YOLO(r"C:\Users\USER\Downloads\best.pt")
#
# # Define the standard size for resizing
# standard_size = (240, 240)  # (width, height)
#
# # Streamlit app title
# st.title("YOLO Zebra Crossing Detection")
#
# # Upload image
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
#
# if uploaded_file is not None:
#     # Read the uploaded image
#     image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
#
#     # Resize the image to the standard size
#     resized_image = cv2.resize(image, standard_size)
#
#     # Run predictions on the resized image
#     results = model.predict(source=resized_image, conf=0.65)  # Adjust conf as needed
#
#     # Display predictions
#     for result in results:
#         annotated_img = result.plot()  # Generates an annotated image with predictions
#         # Convert the annotated image to a format Streamlit can display
#         annotated_img_bgr = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)  # Convert from BGR TO RGB
#         st.image(annotated_img_bgr, caption="YOLO Predictions", channels="RGB")
#
#
#
#
# # Add a footer
# st.write("Upload an image to see YOLO predictions!")
#
# # for name, param in model.model.named_parameters():
# #     print(f"Layer: {name}, Shape: {param.shape}, Weights: {param.data}")
#

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Load the YOLO model
model = YOLO(r"C:\Users\USER\Downloads\best.pt")

# Streamlit app title
st.title("YOLO Zebra Crossing Detection - Video Input")

# Upload video
uploaded_video = st.file_uploader("Upload a video...", type=["mp4", "avi", "mov", "mkv"])

if uploaded_video is not None:
    # Save uploaded video to a temporary file
    temp_file_path = "temp_video.mp4"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_video.read())

    # Open video file
    video_capture = cv2.VideoCapture(temp_file_path)

    # Streamlit placeholder for displaying the video
    stframe = st.empty()

    # Loop through video frames
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        # Resize the frame
        resized_frame = cv2.resize(frame, (640, 640))  # Resize to standard YOLO input size

        # Run YOLO predictions
        results = model.predict(source=resized_frame, conf=0.65)  # Adjust conf threshold as needed

        # Annotate the frame with predictions
        annotated_frame = results[0].plot()

        # Convert BGR to RGB for Streamlit display
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Update the video frame in Streamlit
        stframe.image(annotated_frame_rgb, channels="RGB")

    video_capture.release()

# Footer
st.write("Upload a video to see YOLO predictions in action!")
