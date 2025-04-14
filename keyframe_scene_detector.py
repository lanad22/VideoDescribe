import os
import argparse
import json
import subprocess
import glob
import shutil
import math

import numpy as np
import torch
import clip
from PIL import Image

def get_video_info(video_path):
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    fps_output = subprocess.check_output(cmd).decode().strip()
    num, den = fps_output.split('/')
    fps = float(num) / float(den)
    
    cmd2 = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-count_frames",
        "-show_entries", "stream=nb_read_frames",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    total_frames_str = subprocess.check_output(cmd2).decode().strip()
    try:
        total_frames = int(total_frames_str)
    except Exception as e:
        print(f"Error parsing total frames: {e}")
        total_frames = None
    return fps, total_frames

def extract_frames_ffmpeg(video_path, output_folder, sample_rate=1):
    os.makedirs(output_folder, exist_ok=True)
    command = [
        "ffmpeg", "-i", video_path,
        "-vf", f"select='not(mod(n\\,{sample_rate}))'",
        "-vsync", "vfr",
        "-q:v", "2",
        os.path.join(output_folder, "frame_%06d.jpg")
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print("Error extracting frames:", e.stderr.decode())
        return []
    
    frame_files = sorted(glob.glob(os.path.join(output_folder, "frame_*.jpg")))
    print(f"Extracted {len(frame_files)} frames to {output_folder}")
    return frame_files

def cosine_similarity(emb1, emb2):
    return torch.nn.functional.cosine_similarity(emb1, emb2).item()

def detect_keyframes_and_scene_boundaries(embeddings, keyframe_threshold, scene_boundary_threshold):
    if not embeddings:
        return [], []
    
    keyframes = [0]  # Always mark the first frame as candidate keyframe
    scene_boundaries = [0]  # And as a scene boundary
    last_scene_boundary_index = 0

    for i in range(1, len(embeddings)):
        sim_prev = cosine_similarity(embeddings[i], embeddings[i-1])
        if sim_prev < keyframe_threshold:
            keyframes.append(i)
            sim_scene = cosine_similarity(embeddings[i], embeddings[last_scene_boundary_index])
            if sim_scene < scene_boundary_threshold:
                scene_boundaries.append(i)
                last_scene_boundary_index = i

    return keyframes, scene_boundaries

def segment_video_indices(scene_boundaries, total_frames):
    segments = []
    for i in range(len(scene_boundaries) - 1):
        start = scene_boundaries[i]
        end = scene_boundaries[i+1] - 1
        segments.append((start, end))
    if scene_boundaries and scene_boundaries[-1] < total_frames:
        segments.append((scene_boundaries[-1], total_frames - 1))
    print(f"Segmented video into {len(segments)} raw scenes based on scene boundaries")
    return segments

def adjust_scene_segments_target_duration(segments, fps, target_duration):
    merged_segments = []
    i = 0
    while i < len(segments):
        start, end = segments[i]
        seg_duration = (end - start + 1) / fps
        while seg_duration < target_duration and i < len(segments) - 1:
            i += 1
            next_start, next_end = segments[i]
            end = next_end
            seg_duration = (end - start + 1) / fps
        merged_segments.append((start, end))
        i += 1
    return merged_segments

def extract_video_segment_ffmpeg(video_path, start_time, end_time, output_path):
    duration = end_time - start_time
    command = [
        "ffmpeg", "-y",
        "-ss", str(start_time),
        "-i", video_path,
        "-t", str(duration),
        "-c", "copy",
        output_path
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Extracted scene segment: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error extracting segment: {e.stderr.decode()}")

def process_video_folder(video_folder, sample_rate, keyframe_threshold, scene_boundary_threshold, target_duration, device="cuda"):
    video_id = os.path.basename(os.path.normpath(video_folder))
    video_path = os.path.join(video_folder, f"{video_id}.mp4")
    fps, total_frames = get_video_info(video_path)
    print(f"Processing video: {video_path}")
    print(f"FPS: {fps:.2f}, Total frames: {total_frames}")
    
    # Create a temporary folder for frame extraction.
    temp_folder = os.path.join(video_folder, "frames_temp")
    frame_files = extract_frames_ffmpeg(video_path, temp_folder, sample_rate=sample_rate)
    if not frame_files:
        print("No frames extracted.")
        return None

    # Load CLIP model.
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    embeddings = []
    for frame_file in frame_files:
        try:
            image = Image.open(frame_file).convert("RGB")
        except Exception as e:
            print("Error loading image:", e)
            continue
        image_input = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model.encode_image(image_input)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        embeddings.append(emb)
    print(f"Computed embeddings for {len(embeddings)} frames.")

    # Detect candidate keyframes and scene boundaries using fixed thresholds.
    keyframes, scene_boundaries = detect_keyframes_and_scene_boundaries(embeddings, keyframe_threshold, scene_boundary_threshold)
    print(f"Detected {len(keyframes)} candidate keyframes and {len(scene_boundaries)} scene boundaries.")

    # Save candidate keyframe and scene boundary images if requested.
    keyframes_dir = os.path.join(video_folder, "keyframes")
    os.makedirs(keyframes_dir, exist_ok=True)
    keyframe_info = []
    for idx in keyframes:
        src = frame_files[idx]
        dst = os.path.join(keyframes_dir, f"keyframe_{idx:06d}.jpg")
        shutil.copy2(src, dst)
        timestamp = idx / fps
        keyframe_info.append({
            "frame_index": idx,
            "timestamp": round(timestamp, 2),
            "image_path": os.path.join(video_folder, "keyframes", f"keyframe_{idx:06d}.jpg")
        })
    keyframes_json_path = os.path.join(keyframes_dir, "keyframe_info.json")
    with open(keyframes_json_path, "w") as f:
        json.dump(keyframe_info, f, indent=2)
    print(f"Candidate keyframe info saved to: {keyframes_json_path}")

    scene_dir = os.path.join(video_folder, "scene_boundaries")
    os.makedirs(scene_dir, exist_ok=True)
    for idx in scene_boundaries:
        src = frame_files[idx]
        dst = os.path.join(scene_dir, f"scene_boundary_{idx:06d}.jpg")
        shutil.copy2(src, dst)
    print(f"Scene boundary images saved to: {scene_dir}")

    # Remove temporary folder.
    shutil.rmtree(temp_folder)

    # Segment the video using the detected scene boundaries.
    segments = segment_video_indices(scene_boundaries, total_frames)
    
    # Optionally adjust segments towards a target duration if desired.
    if target_duration > 0:
        segments = adjust_scene_segments_target_duration(segments, fps, target_duration)
    
    print(f"Scene segmentation resulted in {len(segments)} segments before final merging check.")
    
    # ----- Merge the last scene if it is less than 5 seconds -----
    if len(segments) > 1:
        last_segment = segments[-1]
        last_duration = (last_segment[1] - last_segment[0] + 1) / fps
        if last_duration < 5.0:
            print(f"Last segment duration ({last_duration:.2f}s) is less than 5s. Merging it with the previous scene.")
            # Merge with previous segment by extending its end_frame to the last segment's end_frame.
            prev_segment = segments[-2]
            merged_segment = (prev_segment[0], last_segment[1])
            segments[-2] = merged_segment
            segments.pop()
    # ----- End merge block -----
    
    print(f"Final scene segmentation: {len(segments)} segments")

    # Create a folder to store scene segments.
    scenes_dir = os.path.join(video_folder, f"{video_id}_scenes")
    os.makedirs(scenes_dir, exist_ok=True)
    
    # Extract each scene using ffmpeg.
    scene_info = []
    for i, (start_frame, end_frame) in enumerate(segments):
        start_time = start_frame / fps
        # For the last scene, extract until the end of the video.
        end_time = (end_frame / fps) if i < len(segments) - 1 else (total_frames / fps)
        duration = end_time - start_time
        scene_filename = f"scene_{i+1:03d}.mp4"
        scene_path = os.path.join(scenes_dir, scene_filename)
        
        print(f"\nScene {i+1}: frames {start_frame} to {end_frame}, time {start_time:.2f}s to {end_time:.2f}s (duration: {duration:.2f}s)")
        extract_video_segment_ffmpeg(video_path, start_time, end_time, scene_path)
        
        scene_dict = {
            "scene_number": i + 1,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "scene_path": scene_path
        }
        scene_info.append(scene_dict)
    
    # Save scene segmentation info to JSON.
    scenes_json_path = os.path.join(scenes_dir, "scene_info.json")
    with open(scenes_json_path, "w") as jf:
        json.dump(scene_info, jf, indent=2)
    print(f"\nScene processing complete! JSON info saved to: {scenes_json_path}")

    return keyframes, scene_boundaries, fps, total_frames

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Threshold Keyframe and Scene Boundary Detection with Video Segmentation using CLIP.\n")
    parser.add_argument("video_folder", type=str, help="Path to the video folder (e.g., videos/video_id)")
    parser.add_argument("--sample_rate", type=int, default=1, help="Extract every nth frame (default: 1)")
    parser.add_argument("--keyframe_threshold", type=float, default=0.95, help="Cosine similarity threshold for candidate keyframes (default: 0.95)")
    parser.add_argument("--scene_boundary_threshold", type=float, default=0.80, help="Cosine similarity threshold for scene boundaries (default: 0.80)")
    parser.add_argument("--target_duration", type=float, default=10.0, help="Target scene duration in seconds (default: 10)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run CLIP (default: cuda)")
    args = parser.parse_args()
    
    result = process_video_folder(
        args.video_folder, 
        args.sample_rate, 
        args.keyframe_threshold, 
        args.scene_boundary_threshold, 
        args.target_duration, 
        device=args.device
    )
