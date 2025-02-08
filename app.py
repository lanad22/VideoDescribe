import os
import cv2
import torch
from flask import Flask, render_template, jsonify, request, send_from_directory
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image

# ***** Load AI Model *****
print("Loading model... (this may take a while)")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype=torch.float16, device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
print("Model loaded.")

# Flask App Setup
app = Flask(__name__)

VIDEO_PATH = "/home/do.ng/VideoDescribe/videos/n9nC8liwZ5Y/n9nC8liwZ5Y_scenes/scene_002.mp4"

# ***** Serve Video from Local Directory *****
@app.route("/video")
def serve_video():
    if not os.path.exists(VIDEO_PATH):
        return "Error: Video file not found.", 404
    return send_from_directory(os.path.dirname(VIDEO_PATH), os.path.basename(VIDEO_PATH))

# ***** Extract a Frame at Given Timestamp *****
def extract_frame(video_path, timestamp):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Error: Cannot open video file."

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        return None, "Error: FPS is 0, video may be corrupted."

    target_frame = int(timestamp * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    cap.release()

    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)

        # Resize if too large
        max_size = 800
        width, height = image.size
        if max(width, height) > max_size:
            scaling_factor = max_size / max(width, height)
            new_size = (int(width * scaling_factor), int(height * scaling_factor))
            image = image.resize(new_size, Image.ANTIALIAS)
        
        return image, None
    else:
        return None, "Error: Could not extract frame at the given timestamp."

# ***** API Route to Process Frame and Generate Caption *****
@app.route("/process", methods=["POST"])
def process_frame():
    data = request.json
    timestamp = data.get("timestamp")
    question = data.get("question", "Describe this image.")

    if timestamp is None:
        return jsonify({"error": "Missing timestamp"}), 400

    frame, error = extract_frame(VIDEO_PATH, timestamp)
    if error:
        return jsonify({"error": error}), 500

    torch.cuda.empty_cache()  # Free GPU memory

    try:
        inputs = processor(images=frame, text=question, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs)
        answer = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ***** Route to Render UI *****
@app.route("/")
def index():
    return render_template("index.html")

# Run Flask App
if __name__ == "__main__":
    app.run(debug=True)
