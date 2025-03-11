import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO


# Load the YOLO model
model = YOLO(r"C:\Users\USER\Downloads\best.pt")

# Define the standard size for resizing
standard_size = (240, 240)  # (width, height)

# Streamlit app title
st.title("YOLO Zebra Crossing Detection")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Resize the image to the standard size
    resized_image = cv2.resize(image, standard_size)

    # Run predictions on the resized image
    results = model.predict(source=resized_image, conf=0.65)  # Adjust conf as needed

    # Display predictions
    for result in results:
        annotated_img = result.plot()  # Generates an annotated image with predictions
        # Convert the annotated image to a format Streamlit can display
        annotated_img_bgr = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)  # Convert from BGR TO RGB
        st.image(annotated_img_bgr, caption="YOLO Predictions", channels="RGB")




# Add a footer
st.write("Upload an image to see YOLO predictions!")
