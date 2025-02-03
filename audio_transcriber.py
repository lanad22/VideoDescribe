import argparse
import os
import json
import whisper
from pathlib import Path

def transcribe_scenes(scenes_dir: str, model_name: str):
    """
    Loads Whisper model and transcribes each scene video file in the given directory. 
    The transcript is saved as a text file (with the same basename) and its entry is updated with a new key
    "audio_transcript_path" in scene_info.json

    Args:
        scenes_dir (str): Directory containing scene MP4 video files (and optionally scene_info.json).
        model_name (str): Whisper model to use (e.g., tiny, base, small, medium, large).
    """
    print(f"Loading Whisper model '{model_name}'...")
    model = whisper.load_model(model_name)

    scenes_path = Path(scenes_dir)
    
    # Load scene_info.json
    scene_info_file = scenes_path / "scene_info.json"
    if scene_info_file.exists():
        with open(scene_info_file, "r", encoding="utf-8") as f:
            scene_info = json.load(f)
        print(f"Loaded scene_info.json with {len(scene_info)} entries.")
    else:
        scene_info = None
        print("No scene_info.json file found; run scene_detector.py first")

    # Get all MP4 files 
    scene_files = sorted(scenes_path.glob("*.mp4"))
    
    if not scene_files:
        print(f"No MP4 files found in directory: {scenes_dir}; run scene_detector.py first")
        return

    for scene_file in scene_files:
        print(f"\nTranscribing scene: {scene_file.name}")
        try:
            result = model.transcribe(str(scene_file))
        except Exception as e:
            print(f"Error transcribing {scene_file.name}: {e}")
            continue
        
        transcript_text = result.get("text", "").strip()
        transcript_file = scene_file.with_suffix(".txt")
        
        with open(transcript_file, "w", encoding="utf-8") as f:
            f.write(transcript_text)
        
        print(f"Saved transcript to {transcript_file}")

        for scene in scene_info:
            video_path = scene.get("video_path", "")
            if Path(video_path).name == scene_file.name:
                audio_transcript_path = str(Path(video_path).with_suffix(".txt"))
                scene["audio_transcript_path"] = audio_transcript_path
                print(f"Updated scene_info entry for {scene_file.name} with audio_transcript_path: {audio_transcript_path}")
                break

    with open(scene_info_file, "w", encoding="utf-8") as f:
        json.dump(scene_info, f, indent=2)
    print(f"\nUpdated scene_info.json saved to: {scene_info_file.resolve()}")

def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio for each scene video using OpenAI Whisper and update scene_info.json."
    )
    parser.add_argument("scenes_dir", help="Directory containing scene MP4 video files (and optionally scene_info.json).")
    parser.add_argument(
        "--model",
        type=str,
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model to use (default: base)"
    )
    args = parser.parse_args()
    
    transcribe_scenes(args.scenes_dir, args.model)

if __name__ == "__main__":
    main()
