import os
import json
import argparse
import cv2
import whisper
import subprocess
from pathlib import Path
from scenedetect import detect, ContentDetector
from typing import Optional, List, Dict


class VideoProcessor:
    def __init__(self, min_scene_duration=5.0, whisper_model="base"):
        self.min_scene_duration = min_scene_duration
        self.whisper_model_name = whisper_model
        self.detector = None
        self.whisper_model = None

    def extract_scene_with_audio(self, video_path: str, start_frame: int, end_frame: int, output_path: str):
        """Extract scene preserving original audio quality."""
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
            "-c", "copy",  # Copy without re-encoding
            output_path
        ]
        print(f"Extracting scene with audio: {' '.join(command)}")
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

    def check_video_has_speech(self, video_path: str) -> tuple[bool, bool]:
        """Check if video has audio and speech."""
        command = [
            "ffprobe", 
            "-i", video_path,
            "-show_streams", 
            "-select_streams", "a", 
            "-loglevel", "error"
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True)
            has_audio = bool(result.stdout.strip())
            if not has_audio:
                return False, False
        except subprocess.CalledProcessError:
            return False, False

        # Load Whisper model if needed
        if not self.whisper_model:
            print(f"Loading Whisper model '{self.whisper_model_name}'...")
            self.whisper_model = whisper.load_model(self.whisper_model_name)

        try:
            # Use a small segment for quick speech detection
            temp_path = video_path + ".temp.wav"
            extract_command = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-t", "10",  # Check first 10 seconds
                "-vn",  # No video
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                temp_path
            ]
            subprocess.run(extract_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            result = self.whisper_model.transcribe(temp_path)
            os.remove(temp_path)

            transcript = result.get("text", "").strip()
            has_speech = bool(transcript) and len(transcript) > 5
            return True, has_speech

        except Exception as e:
            print(f"Error during speech detection: {e}")
            return True, False

    def process_video(self, video_path: str, captions_path: Optional[str] = None) -> List[Dict]:
        """Process video: detect scenes, generate transcripts, and prepare for Qwen."""
        print(f"Processing video: {video_path}")

        # Check for audio and speech
        has_audio, has_speech = self.check_video_has_speech(video_path)
        print(f"Video has audio: {has_audio}")
        print(f"Video has speech: {has_speech}")

        video_id = Path(video_path).stem
        output_dir = os.path.join(os.path.dirname(video_path), f"{video_id}_scenes")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Load captions if available
        captions = None
        if captions_path and os.path.exists(captions_path):
            print("Loading captions...")
            with open(captions_path, 'r') as f:
                captions = json.load(f)

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

            # First extract with audio (for transcription and as backup)
            scene_with_audio = os.path.join(output_dir, f"scene_{i+1:03d}_with_audio.mp4")
            self.extract_scene_with_audio(video_path, start_frame, end_frame, scene_with_audio)

            # Create Qwen-compatible version
            scene_for_qwen = os.path.join(output_dir, f"scene_{i+1:03d}.mp4")
            self.convert_for_qwen(scene_with_audio, scene_for_qwen)

            info = {
                'scene_number': i + 1,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration,
                'video_path': scene_for_qwen
                #'has_audio': has_audio,
                #'has_speech': has_speech
            }

            # Handle transcription if speech is detected
            if has_speech:
                print(f"Transcribing audio for scene {i+1}...")
                transcript_path = os.path.join(output_dir, f"scene_{i+1:03d}.txt")
                try:
                    result = self.whisper_model.transcribe(scene_with_audio)
                    transcript_text = result.get("text", "").strip()
                    
                    with open(transcript_path, "w", encoding="utf-8") as f:
                        f.write(transcript_text)
                    
                    info['transcript_path'] = transcript_path
                    #info['transcript'] = transcript_text
                    print(f"Saved transcript to {transcript_path}")
                except Exception as e:
                    print(f"Error during transcription: {e}")

            # Delete the audio version after transcription
            os.remove(scene_with_audio)

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
        print(f"Results saved to: {output_dir}")
        print(f"Scene information saved to: {info_path}")

        return scene_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process video: detect scenes, generate transcripts, and prepare for Qwen"
    )
    parser.add_argument("video_path", 
                        help="Path to the input video file")
    parser.add_argument("--whisper-model",
                        type=str,
                        default="base",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model to use (default: base)")
    parser.add_argument("--captions",
                        help="Path to the captions file (optional)",
                        default=None)
    parser.add_argument("--min-duration",
                        type=float,
                        default=5.0,
                        help="Minimum scene duration in seconds (default: 5.0)")
    args = parser.parse_args()

    processor = VideoProcessor(min_scene_duration=args.min_duration,
                             whisper_model=args.whisper_model)
    processor.process_video(args.video_path, args.captions)