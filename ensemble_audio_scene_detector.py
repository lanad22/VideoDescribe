import os
import sys
import json
import argparse
import cv2
import subprocess
import whisper_timestamped
import torch
import librosa
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from pathlib import Path
from scenedetect import detect, ContentDetector
from typing import Optional

import warnings
import onnxruntime

onnxruntime.set_default_logger_severity(3) 
os.environ["OMP_NUM_THREADS"] = "1" 

class SceneDetector:
    def __init__(self, min_scene_duration=10.0, whisper_model="large-v2", wav2vec_model="facebook/wav2vec2-large-960h-lv60-self", device="cuda"):
        self.min_scene_duration = min_scene_duration
        self.detector = None
        self.whisper_model_name = whisper_model
        self.wav2vec_model_name = wav2vec_model
        self.device = device
        self.whisper_model = None 
        self.wav2vec_model = None  
        self.wav2vec_processor = None  

    def extract_scene(self, video_folder: str, start_frame: int, end_frame: int, output_path: str):
        video_id = os.path.basename(os.path.normpath(video_folder))
        video_path = os.path.join(video_folder, f"{video_id}.mp4")
        if not os.path.exists(video_path):
            print(f"Error: Video file not found at {video_path}")
            return

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        start_time = start_frame / fps
        duration = (end_frame - start_frame) / fps

        # Changed FFmpeg command to fix black frames issue
        # - Moved -ss before input for precise seeking
        # - Using re-encode instead of stream copy
        command = [
            "ffmpeg", "-y",
            "-ss", str(start_time),
            "-i", video_path,
            "-t", str(duration),
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "18",
            "-force_key_frames", "expr:gte(t,0)",
            "-c:a", "aac",
            output_path
        ]
        print(f"Extracting scene: {' '.join(command)}")
        try:
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print(f"Error during scene extraction:", e.stderr.decode())
            raise

    def extract_audio(self, video_path, output_path):
        """Extract audio from video file."""
        command = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vn",  # Disable video
            "-acodec", "pcm_s16le",  # Audio codec
            "-ar", "16000",  # Sample rate for wav2vec2
            "-ac", "1",  # Mono audio
            output_path
        ]
        print(f"Extracting audio: {' '.join(command)}")
        try:
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print(f"Error during audio extraction:", e.stderr.decode())
            raise

    def transcribe_with_wav2vec(self, audio_path):
        """Transcribe audio using wav2vec2."""
        print(f"Transcribing with wav2vec2: {audio_path}")

        # Load the model
        if self.wav2vec_model is None:
            print(f"Loading wav2vec2 model: {self.wav2vec_model_name}")
            self.wav2vec_processor = Wav2Vec2Processor.from_pretrained(self.wav2vec_model_name)
            self.wav2vec_model = Wav2Vec2ForCTC.from_pretrained(self.wav2vec_model_name).to(self.device)

        try:
            # Load audio
            speech_array, _ = librosa.load(audio_path, sr=16000)

            # Process in chunks dynamically based on silence detection
            chunk_size = 16000 * 60  # 60 seconds per chunk
            transcripts = []

            for i in range(0, len(speech_array), chunk_size):
                chunk = speech_array[i:min(i + chunk_size, len(speech_array))]

                # Process audio
                inputs = self.wav2vec_processor(chunk, sampling_rate=16000, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    logits = self.wav2vec_model(inputs.input_values).logits

                # Decode
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = self.wav2vec_processor.batch_decode(predicted_ids)[0].lower()  # Convert to lowercase

                # Calculate approximate timestamp
                start_time = i / 16000  # in seconds
                end_time = min(i + chunk_size, len(speech_array)) / 16000

                transcripts.append({
                    "text": transcription.strip(),
                    "start": start_time,
                    "end": end_time
                })

            print(f"Wav2vec transcription successful: {len(transcripts)} segments")
            return transcripts

        except Exception as e:
            print(f"Error during wav2vec transcription: {str(e)}")
            return []

    def transcribe_scene(self, scene_path: str):
        """Transcribe a scene using Whisper with timestamps relative to the scene"""
        print(f"Transcribing scene with Whisper: {scene_path}")
        
        # Load the model
        if self.whisper_model is None:
            print(f"Loading whisper model: {self.whisper_model_name}")
            self.whisper_model = whisper_timestamped.load_model(self.whisper_model_name, device=self.device)
        
        try:
            result = whisper_timestamped.transcribe(
                self.whisper_model,
                scene_path,
                vad="silero:v3.1",
                beam_size=10, 
                best_of=5, 
                temperature=(0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0)
            )
            
            # Convert to our desired format
            transcripts = []
            for segment in result["segments"]:
                transcripts.append({
                    "text": segment["text"],
                    "start": segment["start"],
                    "end": segment["end"]
                })
            
            print(f"Whisper transcription successful: {len(transcripts)} segments")
            return transcripts
        except Exception as e:
            print(f"Error during Whisper transcription: {str(e)}")
            return []

    def verify_transcriptions(self, whisper_transcripts, wav2vec_transcripts, threshold=0.6):
        """Verify whisper transcriptions against wav2vec to reduce hallucinations."""
        if not wav2vec_transcripts:
            return whisper_transcripts  # If wav2vec failed, fall back to whisper

        def text_similarity(text1, text2):
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            return len(intersection) / len(union) if union else 0

        verified_transcripts = []

        for whisper_segment in whisper_transcripts:
            overlapping_segments = []
            for wav2vec_segment in wav2vec_transcripts:
                # Check if segments overlap in time
                if (whisper_segment["start"] <= wav2vec_segment["end"] and
                    whisper_segment["end"] >= wav2vec_segment["start"]):
                    overlapping_segments.append(wav2vec_segment)

            if not overlapping_segments:
                # No overlap found, check if Whisper is hallucinating
                if len(whisper_segment["text"].strip()) == 1:  # Likely an emoji or gibberish
                    continue  # Remove this hallucinated text
                whisper_segment["verified"] = False
                verified_transcripts.append(whisper_segment)
                continue

            # Combine overlapping wav2vec segments
            wav2vec_text = " ".join([seg["text"] for seg in overlapping_segments]).strip()

            # Check if Wav2Vec2 is empty
            if not wav2vec_text:
                continue  # Remove Whisper segment if Wav2Vec2 has no content

            # Calculate similarity
            similarity = text_similarity(whisper_segment["text"], wav2vec_text)

            '''
            whisper_segment["verified"] = similarity >= threshold
            whisper_segment["verification_score"] = similarity'''
            verified_transcripts.append(whisper_segment)

        return verified_transcripts

    def detect_scenes(self, video_folder: str, captions_path: Optional[str] = None) -> str:
        """Detects scenes and transcribes using Whisper + Wav2Vec2, verifying hallucinations, and matching with captions."""
        video_id = os.path.basename(os.path.normpath(video_folder))
        video_path = os.path.join(video_folder, f"{video_id}.mp4")
        print(f"Processing video: {video_path}")

        # Create directory for extracted scenes
        scenes_dir = os.path.join(video_folder, f"{video_id}_scenes")
        os.makedirs(scenes_dir, exist_ok=True)

        # Load captions if available
        captions = None
        captions_path = os.path.join(video_folder, f"{video_id}.json")

        if captions_path and os.path.exists(captions_path):
            print("Loading captions...")
            with open(captions_path, 'r') as f:
                captions_data = json.load(f)
            
            if "captions" in captions_data and captions_data["captions"]:
                captions = captions_data["captions"]
            else:
                print("No captions found in the JSON file.")

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        min_scene_frames = int(self.min_scene_duration * fps)
        self.detector = ContentDetector(threshold=35.0, min_scene_len=min_scene_frames)

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

            # Extract scene
            scene_path = os.path.join(scenes_dir, f"scene_{i+1:03d}.mp4")
            self.extract_scene(video_folder, start_frame, end_frame, scene_path)

            # Extract audio
            audio_path = os.path.join(scenes_dir, f"scene_{i+1:03d}.wav")
            self.extract_audio(scene_path, audio_path)

            # Transcribe using Whisper and Wav2Vec2
            whisper_transcripts = self.transcribe_scene(scene_path)
            wav2vec_transcripts = self.transcribe_with_wav2vec(audio_path)

            # Verify transcripts (remove hallucinations)
            verified_transcripts = self.verify_transcriptions(whisper_transcripts, wav2vec_transcripts)

            info = {
                'scene_number': i + 1,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration,
                'scene_path': scene_path,
                'transcripts': verified_transcripts
            }

            # Add matching captions if available
            if captions:
                scene_captions = []
                for caption in captions:
                    cap_start = caption['start']
                    cap_duration = caption['duration']
                    cap_end = cap_start + cap_duration
                    
                    # Check if caption overlaps substantially with this scene
                    if cap_start < end_time and cap_end > start_time:
                        # Calculate overlap amount
                        overlap_start = max(cap_start, start_time)
                        overlap_end = min(cap_end, end_time)
                        overlap_duration = overlap_end - overlap_start
                        
                        # Only include caption if at least 50% of it is in this scene
                        if overlap_duration >= (cap_duration * 0.5):
                            scene_captions.append({
                                'text': caption['text'],
                                'start': max(cap_start - start_time, 0),  # Relative to scene start
                                'end': min(cap_end - start_time, duration)  # Relative to scene start
                            })
                
                if scene_captions:
                    info['captions'] = scene_captions

            scene_info.append(info)

        # Save results to JSON file
        info_path = os.path.join(scenes_dir, 'scene_info.json')
        with open(info_path, 'w') as f:
            json.dump(scene_info, f, indent=2)

        print(f"\nScene detection and transcription complete!")
        print(f"Total scenes: {len(scene_info)}")
        print(f"Results saved to: {info_path}")

        return info_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect scenes and transcribe with Whisper and wav2vec2"
    )
    
    parser.add_argument("video_folder", type=str,
                        help="Folder containing video files (format: folder/video_id/video_id.mp4)")
    parser.add_argument("--min-duration",
                        type=float,
                        default=7.0,
                        help="Minimum scene duration in seconds (default: 7.0)")
    parser.add_argument("--whisper-model",
                        type=str,
                        default="large-v2",
                        help="Whisper model to use (default: large-v2)")
    
    args = parser.parse_args()

    detector = SceneDetector(
        min_scene_duration=args.min_duration,
        whisper_model=args.whisper_model,
    )
    
    # Detect scenes and transcribe
    detector.detect_scenes(args.video_folder)