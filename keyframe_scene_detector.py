import os
import argparse
import json
import subprocess
import numpy as np
import cv2
from decord import VideoReader, cpu

def extract_keyframes(video_path):
    """
    Extract only keyframe indices from the video using decord.
    """
    print(f"Extracting keyframes from {video_path}")
    try:
        # Single-threaded decoding to resolve the packet sending error
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total_frames = len(vr)
        print(f"Total frames: {total_frames} frames")
        
        # Get keyframe indices - the only thing we actually need
        keyframe_indices = vr.get_key_indices()
        print(f"Found {len(keyframe_indices)} keyframe indices")
        
        return keyframe_indices, total_frames
    except Exception as e:
        print(f"Error extracting keyframes: {str(e)}")
        return [], 0

def segment_video_indices(keyframe_indices, total_frames):
    """
    Create a list of (start_frame, end_frame) pairs for each segment.
    Each segment goes from one keyframe to one frame before the next.
    """
    segments = []
    for i in range(len(keyframe_indices) - 1):
        start = keyframe_indices[i]
        end = keyframe_indices[i+1] - 1  # end at the frame before next keyframe
        segments.append((start, end))
    if scene_boundaries and scene_boundaries[-1] < total_frames:
        segments.append((scene_boundaries[-1], total_frames - 1))
    print(f"Segmented video into {len(segments)} raw scenes based on scene boundaries")
    return segments

def combine_segments(segments, group_size, target_duration, fps):
    combined = []
    num_segments = len(segments)
    i = 0
    while i < num_segments:
        # For the last group, which may not be full:
        if i + group_size >= num_segments:
            group = segments[i:num_segments]
            start = group[0][0]
            end = group[-1][1]
            duration = (end - start + 1) / fps  # calculate duration in seconds
            # If the duration is too short and we already have a previous group, merge them.
            if duration < 0.5 * target_duration and combined:
                prev_start, prev_end = combined.pop()
                new_start = prev_start
                new_end = end
                combined.append((new_start, new_end))
            else:
                combined.append((start, end))
            break
        else:
            group = segments[i:i+group_size]
            start = group[0][0]
            end = group[-1][1]
            combined.append((start, end))
            i += group_size
    print(f"Combined segments into {len(combined)} segments using group size {group_size}")
    return combined

def extract_video_segment_ffmpeg(video_path, start_time, end_time, output_path):
    """
    Use ffmpeg to extract a segment from the original video (with audio).
    Uses stream copy to avoid re-encoding.
    """
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

def process_video(video_folder, target_duration, device="cuda"):
    """
    Process the video using keyframe-based segmentation to define scenes.
    For each scene:
      1. Determine scene boundaries (via keyframe segmentation, optionally combined to ~target_duration seconds).
      2. Extract the original scene (with audio) using ffmpeg.
      3. If a captions JSON file exists, match overlapping captions.
      4. Save scene information (including boundaries, file paths, and captions) to a JSON file.
    """
    video_id = os.path.basename(os.path.normpath(video_folder))
    video_path = os.path.join(video_folder, f"{video_id}.mp4")
    fps, total_frames = get_video_info(video_path)
    print(f"Processing video: {video_path}")
    
    # Extract keyframes and all frames
    keyframe_indices, total_frames = extract_keyframes(video_path)
    if not keyframe_indices or not total_frames:
        print("Failed to extract keyframes/frames. Exiting.")
        return
    
    # Get FPS and total duration
    try:
        vr_temp = VideoReader(video_path, ctx=cpu(0))
        fps = vr_temp.get_avg_fps()
        total_duration = total_frames / fps
        print(f"Video duration: {total_duration:.2f}s, FPS: {fps:.2f}")
    except Exception as e:
        print(f"Error computing video duration: {str(e)}")
        fps = 30.0
        total_duration = total_frames / fps
    
    # Segment the video using keyframe indices
    segments = segment_video_indices(keyframe_indices, total_frames)
    
    # Compute average segment duration and determine group size to reach ~target_duration seconds
    if segments:
        avg_seg_duration = total_duration / len(segments)
        group_size = max(1, int(round(target_duration / avg_seg_duration)))
        print(f"Average segment duration: {avg_seg_duration:.2f}s, grouping segments with group size: {group_size}")
    else:
        group_size = 1
    
    if group_size > 1:
        segments = combine_segments(segments, group_size, target_duration, fps)
    
    scene_info = []
    
    # Process each scene
    for i, (start_frame, end_frame) in enumerate(segments):
        start_time = start_frame / fps
        end_time = (end_frame + 1) / fps  # include last frame
        duration = end_time - start_time
        scene_filename = f"scene_{i+1:03d}.mp4"
        scene_path = os.path.join(scenes_dir, scene_filename)
        
        print(f"\nScene {i+1}: frames {start_frame} to {end_frame}, time {start_time:.2f}s to {end_time:.2f}s (duration: {duration:.2f}s)")
        
        # Extract original scene (with audio)
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
    
    # Save scene information to JSON
    json_path = os.path.join(scenes_dir, "scene_info.json")
    with open(json_path, "w") as jf:
        json.dump(scene_info, jf, indent=2)
    print(f"\nScene processing complete! JSON info saved to: {scenes_json_path}")

    return keyframes, scene_boundaries, fps, total_frames

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect scenes from a video using keyframe segmentation, extract original segments (with audio), "
                    "and save scene information (including captions if available) to a JSON file."
    )
    parser.add_argument("video_folder", type=str,
                        help="Path to the video folder (e.g., videos/video_id). The video file must be named video_id.mp4")
    parser.add_argument("--target_duration", type=float, default=20.0,
                        help="Desired duration (in seconds) for each scene (default: 20s)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run processing (default: cuda)")
    
    args = parser.parse_args()
    process_video(args.video_folder, args.target_duration, args.device)
