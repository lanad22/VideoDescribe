import os
import json
import argparse
import cv2
import subprocess
from pathlib import Path
from scenedetect import detect, ContentDetector
from typing import Optional, List, Dict

class SceneDetector:
    def __init__(self, min_scene_duration=5.0):
        self.min_scene_duration = min_scene_duration
        self.detector = None

    def extract_scene(self, video_folder: str, start_frame: int, end_frame: int, output_path: str):
        """Extract scene."""
        video_id = os.path.basename(video_folder)
        video_path = os.path.join(video_folder, f"{video_id}.mp4")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        start_time = start_frame / fps
        duration = (end_frame - start_frame) / fps

        command = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-ss", str(start_time),
            "-t", str(duration),
            "-c", "copy",
            output_path
        ]
        print(f"Extracting scene: {' '.join(command)}")
        try:
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print(f"Error during scene extraction:", e.stderr.decode())
            raise

    def convert_for_qwen(self, input_path: str, output_path: str):
        """Convert video to Qwen-compatible format."""
        command = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-an",  # No audio
            "-c:v", "libx264",
            "-preset", "medium",
            "-tune", "film",
            "-profile:v", "high",
            "-crf", "23",
            "-vf", "scale='min(1280,iw)':'-2'",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-g", "30",
            "-keyint_min", "30",
            output_path
        ]
        print(f"Converting for Qwen: {' '.join(command)}")
        try:
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print(f"Error during conversion:", e.stderr.decode())
            raise

    def detect_scenes(self, video_folder: str, captions_path: Optional[str] = None) -> str:
        video_id = os.path.basename(video_folder)
        video_path = os.path.join(video_folder, f"{video_id}.mp4")
        print(f"Processing video: {video_path}")

        # Get video_id and create scenes directory
        video_id = os.path.basename(video_folder)
        #video_dir = os.path.dirname(video_path)
        scenes_dir = os.path.join(video_folder, f"{video_id}_scenes")
        os.makedirs(scenes_dir, exist_ok=True)

        # Load captions if available
        captions = None
        
        #video_id.json 
        caption_path = os.path.join(video_folder, f"{video_id}.json")
        if captions_path and os.path.exists(captions_path):
            print("Loading captions...")
            with open(captions_path, 'r') as f:
                captions_data = json.load(f)
            
            # Only process captions if the 'captions' field exists and is not empty
            if "captions" in captions_data and captions_data["captions"]:
                captions = captions_data["captions"]
            else:
                print("No captions found in the JSON file.")

        # Initialize scene detection
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        min_scene_frames = int(self.min_scene_duration * fps)
        self.detector = ContentDetector(threshold=32.0, min_scene_len=min_scene_frames)

        print("Detecting scenes...")
        scene_list = detect(video_path, self.detector)

        scene_info = []
        print("\nProcessing scenes...")

        for i, scene in enumerate(scene_list):
            start_frame = scene[0].get_frames()
            end_frame = scene[1].get_frames()
            start_time = start_frame / fps
            end_time = end_frame / fps
            duration = end_time - start_time

            print(f"\nProcessing scene {i+1}/{len(scene_list)} (duration: {duration:.2f}s)...")

            # Extract scene and convert for Qwen
            scene_path = os.path.join(scenes_dir, f"scene_{i+1:03d}.mp4")
            temp_path = scene_path + ".temp.mp4"
            
            self.extract_scene(video_folder, start_frame, end_frame, temp_path)
            self.convert_for_qwen(temp_path, scene_path)
            os.remove(temp_path)

            info = {
                'scene_number': i + 1,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration,
                'scene_path': scene_path
            }

            # Add captions if available
            if captions:
                scene_captions = []
                for caption in captions:
                    cap_start = caption['start']
                    cap_end = cap_start + caption['duration']
                    if cap_start <= end_time and cap_end >= start_time:
                        scene_captions.append({
                            'text': caption['text'],
                            'start': max(cap_start - start_time, 0),  # Relative to scene start
                            'end': min(cap_end - start_time, duration)
                        })
                if scene_captions:
                    info['captions'] = scene_captions

            scene_info.append(info)
        
        info_path = os.path.join(scenes_dir, 'scene_info.json')
        with open(info_path, 'w') as f:
            json.dump(scene_info, f, indent=2)

        print(f"\nScene detection complete!")
        print(f"Total scenes: {len(scene_info)}")
        print(f"Results saved to: {info_path}")

        return info_path

    def match_transcripts_to_scenes(self, video_folder: str) -> None:
        """Match transcribed segments to scenes."""
        print("\nMatching transcripts to scenes...")
        
        # Get paths based on video directory
        #video_id = os.path.basename(os.path.dirname(video_path))
        #video_dir = os.path.dirname(video_path)
        
        video_id = os.path.basename(video_folder)
        video_path = os.path.join(video_folder, f"{video_id}.mp4")
        scenes_dir = os.path.join(video_folder, f"{video_id}_scenes")
        
        transcript_path = os.path.join(video_folder, "transcript.json")
        scene_info_path = os.path.join(scenes_dir, "scene_info.json")
        
        # Check if required files exist
        if not os.path.exists(transcript_path):
            print(f"Transcript file not found: {transcript_path}")
            return
        if not os.path.exists(scene_info_path):
            print(f"Scene info file not found: {scene_info_path}")
            return
        
        # Load transcripts and scene info
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcribed_segments = json.load(f)
        
        with open(scene_info_path, 'r') as f:
            scene_info = json.load(f)
        
        # Match segments to scenes
        for scene in scene_info:
            scene_start = scene['start_time']
            scene_end = scene['end_time']
            scene_transcripts = []
            
            # Find all segments that overlap with this scene
            for segment in transcribed_segments:
                seg_start = segment['start_time']
                seg_end = segment['end_time']
                
                # Check if segment overlaps with scene
                if (seg_start <= scene_end) and (seg_end >= scene_start):
                    scene_transcripts.append({
                        'text': segment['text'],
                        'start': max(seg_start - scene_start, 0),  # Relative to scene start
                        'end': min(seg_end - scene_start, scene_end - scene_start)
                    })
            
            # Add transcripts to scene info
            if scene_transcripts:
                scene['transcripts'] = scene_transcripts
        
        # Update scene info file with transcripts
        with open(scene_info_path, 'w') as f:
            json.dump(scene_info, f, indent=2)
        
        print("Transcript matching complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect scenes and match with transcripts"
    )
    #parser.add_argument("video_path", help="Path to the input video file (format: videos/video_id/video_id.mp4)")
    
    parser.add_argument("video_folder", type=str,
                        help="Folder containing files associated video files.")
    parser.add_argument("--min-duration",
                        type=float,
                        default=5.0,
                        help="Minimum scene duration in seconds (default: 5.0)")
    args = parser.parse_args()

    # Initialize and run scene detection
    detector = SceneDetector(min_scene_duration=args.min_duration)
    
    # First detect scenes
    detector.detect_scenes(args.video_folder)
    
    # Then match with transcripts
    detector.match_transcripts_to_scenes(args.video_folder)