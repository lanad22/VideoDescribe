import sys
import argparse
import streamlit as st
import cv2
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
import os

# Parse command-line arguments (using parse_known_args to allow Streamlit's own args)
parser = argparse.ArgumentParser()
parser.add_argument("--video_path", type=str, help="Path to the video file")
args, _ = parser.parse_known_args()
video_path_arg = args.video_path if args.video_path else ""

# Load the Qwen2.5-VL model and processor (cached so they load only once)
@st.cache_resource
def load_model():
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    return model, processor

model, processor = load_model()

st.title("Local Video Q&A with Qwen2.5-VL")

# Use the command-line argument (if provided) as the default value for the text input
video_path = st.text_input("Enter the path to your video file:", value=video_path_arg)

if video_path:
    try:
        # Display the video by reading its bytes and using Streamlit's video component
        with open(video_path, "rb") as video_file:
            video_bytes = video_file.read()
        st.video(video_bytes)
    except Exception as e:
        st.error(f"Error loading video: {e}")

    st.write("Play the video and pause at the moment you want to ask a question. Then, note the timestamp (in seconds) and enter your question below:")

    # Get timestamp and question from the user
    timestamp = st.number_input("Enter the timestamp (in seconds):", min_value=0, value=0, step=1)
    question = st.text_input("Enter your question:")

    if st.button("Ask Question") and question:
        st.write("Processing video... This may take a moment.")
        try:
            # Open the video file and calculate the frame corresponding to the timestamp
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            target_frame = int(timestamp * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()
            if not ret:
                st.error("Could not extract frame at the given timestamp.")
            else:
                # Convert the frame from BGR (OpenCV format) to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                st.image(image, caption=f"Frame at {timestamp} seconds")
                
                # Process the image and the user question using the Qwen2.5-VL model
                inputs = processor(images=image, text=question, return_tensors="pt").to(model.device)
                outputs = model.generate(**inputs)
                answer = processor.batch_decode(outputs, skip_special_tokens=True)[0]
                
                st.write("Answer:")
                st.write(answer)
            cap.release()
        except Exception as e:
            st.error(f"An error occurred: {e}")
