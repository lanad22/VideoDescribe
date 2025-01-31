import os
import time
import json
import argparse
import cv2
from pathlib import Path
from scenedetect import detect, ContentDetector, FrameTimecode
from typing import Optional, List, Dict


class PySceneDetector:
    def __init__(self, detection_method="content", min_scene_duration=5.0):
        """
        Initialize scene detector with the specified method and minimum scene duration.
        
        Args:
            detection_method (str): Method used for scene detection (currently only "content" supported)
            min_scene_duration (float): Minimum duration for each scene in seconds
        """
        self.detection_method = detection_method
        self.min_scene_duration = min_scene_duration
        # Configure the detector with min_scene_length based on FPS
        self.detector = None  # Will be initialized when processing video

    def extract_scene_video(self, video_path: str, start_frame: int, end_frame: int, output_path: str):
        """Extract scene to a separate video file."""
        cap = cv2.VideoCapture(video_path)

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Read and write frames
        for _ in range(end_frame - start_frame):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        cap.release()
        out.release()

    def detect_and_save_scenes(self, video_path: str, captions_path: Optional[str] = None) -> List[Dict]:
        """Detect and save scenes, along with matching captions if available."""

        print(f"Processing video: {video_path}")

        # Create output directory for scenes
        video_id = Path(video_path).stem
        output_dir = os.path.join(os.path.dirname(video_path), f"{video_id}_scenes")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Load captions if available
        captions = None
        if captions_path and os.path.exists(captions_path):
            print("Loading captions...")
            with open(captions_path, 'r') as f:
                captions = json.load(f)

        # Get video properties first
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        # Initialize detector with minimum scene length in frames
        min_scene_frames = int(self.min_scene_duration * fps)
        self.detector = ContentDetector(threshold=32.0, min_scene_len=min_scene_frames) #increase threshold from 27 for longer scene

        # Detect scenes
        print("Detecting scenes...")
        scene_list = detect(video_path, self.detector)

        # Process scenes
        scene_info = []
        print("\nExtracting scenes and matching captions...")

        for i, scene in enumerate(scene_list):
            # Get scene boundaries
            start_frame = scene[0].get_frames()
            end_frame = scene[1].get_frames()
            start_time = start_frame / fps
            end_time = end_frame / fps
            duration = end_time - start_time

            # Create scene filename
            scene_filename = f"scene_{i+1:03d}.mp4"
            scene_path = os.path.join(output_dir, scene_filename)

            # Extract and save scene video
            print(f"Extracting scene {i+1}/{len(scene_list)} (duration: {duration:.2f}s)...")
            self.extract_scene_video(video_path, start_frame, end_frame, scene_path)

            info = {
                'scene_number': i + 1,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration,
                'video_path': scene_path
            }

            # Add captions if available
            if captions:
                scene_captions = []
                for caption in captions:
                    cap_start = caption['start']
                    cap_end = cap_start + caption['duration']
                    if cap_start <= end_time and cap_end >= start_time:
                        scene_captions.append(caption['text'])
                info['captions'] = scene_captions

            scene_info.append(info)
        
        info_path = os.path.join(output_dir, 'scene_info.json')
        with open(info_path, 'w') as f:
            json.dump(scene_info, f, indent=2)

        print(f"\nProcessing complete!")
        print(f"Total scenes: {len(scene_info)}")
        print(f"Scenes saved to: {output_dir}")
        print(f"Scene information saved to: {info_path}")

        return scene_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect and save scenes from a video.")
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument("--captions", help="Path to the captions file (optional)", default=None)
    parser.add_argument("--min-duration", type=float, default=5.0, 
                      help="Minimum scene duration in seconds (default: 5.0)")
    args = parser.parse_args()

    detector = PySceneDetector(min_scene_duration=args.min_duration)
    detector.detect_and_save_scenes(args.video_path, args.captions)