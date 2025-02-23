import os
import sys
import argparse
import json
import subprocess
from gtts import gTTS
import shutil

def run_command(cmd):
    """Run a shell command and return its stdout as a string; exit on error."""
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f"Command failed:\n{cmd}\nError: {result.stderr}")
        sys.exit(1)
    return result.stdout.strip()

def get_audio_duration(filepath):
    """Return the duration (in seconds) of an audio file using ffprobe."""
    cmd = f'ffprobe -v error -select_streams a:0 -show_entries format=duration -of csv="p=0" "{filepath}"'
    out = run_command(cmd)
    try:
        return float(out)
    except ValueError:
        print(f"Could not determine audio duration for {filepath}")
        sys.exit(1)

def generate_tts(description, output_audio):
    """Generate a TTS MP3 from the description text and save it to output_audio."""
    tts = gTTS(text=description, lang='en')
    tts.save(output_audio)
    print(f"Generated TTS: {output_audio}")

def find_insertion_gap(transcripts, tts_duration, scene_duration):
    """Find a gap in the transcripts for inserting TTS."""
    segments = sorted(transcripts, key=lambda x: x["start"])
    if segments:
        if segments[0]["start"] >= tts_duration:
            return 0.0
        for i in range(len(segments) - 1):
            gap = segments[i+1]["start"] - segments[i]["end"]
            if gap >= tts_duration:
                return segments[i]["end"]
        if scene_duration - segments[-1]["end"] >= tts_duration:
            return segments[-1]["end"]
    return max(0, scene_duration - tts_duration)

def main():
    parser = argparse.ArgumentParser(
        description="Insert all scene TTS descriptions into the full audio and mux with the full video."
    )
    parser.add_argument("--video_id", type=str, required=True, help="Video ID (folder name)")
    parser.add_argument("--scene", type=int, default=None, help="Optional: process only a specific scene number")
    args = parser.parse_args()
    
    video_id = args.video_id
    base_dir = os.path.join("videos", video_id)
    video_path = os.path.join(base_dir, f"{video_id}.mp4")
    audio_path = os.path.join(base_dir, f"{video_id}.wav")
    json_path = os.path.join(base_dir, f"{video_id}_scenes", "scene_info.json")
    output_video = os.path.join(base_dir, "final_video.mp4")
    
    # verify inputs exist
    for path, desc in [(video_path, "video"), (audio_path, "audio"), (json_path, "JSON")]:
        if not os.path.exists(path):
            print(f"{desc.capitalize()} file not found: {path}")
            sys.exit(1)
    
    # load and process scene information
    with open(json_path, "r") as f:
        scenes = json.load(f)
    
    if args.scene is not None:
        scenes = [s for s in scenes if s.get("scene_number") == args.scene]
        if not scenes:
            print(f"Scene {args.scene} not found in {json_path}")
            sys.exit(1)
    
    scenes.sort(key=lambda s: s.get("start_time", 0))
    
    working_dir = os.path.join(base_dir, "working_temp")
    os.makedirs(working_dir, exist_ok=True)
    
    # Process TTS for each scene
    tts_list = []  # list of tuples: (tts_file, delay_ms)
    for scene in scenes:
        scene_num = scene.get("scene_number")
        description = scene.get("description")
        description_type = scene.get("description_type")
        if not description or description_type != "inline":
            print(f"Scene {scene_num} has no description or description is extended; skipping.")
            continue
        
        tts_file = os.path.join(working_dir, f"tts_scene_{scene_num}.mp3")
        generate_tts(description, tts_file)
        tts_duration = get_audio_duration(tts_file)
        
        scene_start = scene.get("start_time", 0)
        scene_duration = scene.get("duration")
        if scene_duration is None:
            scene_end = scene.get("end_time")
            if scene_end is not None:
                scene_duration = scene_end - scene_start
            else:
                print(f"Scene {scene_num} missing duration/end_time; skipping.")
                continue
        
        transcripts = scene.get("transcripts", [])
        rel_gap = find_insertion_gap(transcripts, tts_duration, scene_duration)
        abs_insertion = scene_start + rel_gap
        delay_ms = int(abs_insertion * 1000)
        
        print(f"Scene {scene_num}: TTS duration = {tts_duration:.2f}s, insertion at = {abs_insertion:.2f}s")
        tts_list.append((tts_file, delay_ms, tts_duration))
    
    # create silent base track
    audio_duration = get_audio_duration(audio_path)
    silent_base = os.path.join(working_dir, "silent_base.wav")
    silence_cmd = f'ffmpeg -y -f lavfi -i anullsrc=r=16000:cl=mono -t {audio_duration} -acodec pcm_s16le "{silent_base}"'
    print("\nCreating silent base track:")
    print(silence_cmd)
    run_command(silence_cmd)
    
    filter_parts = []
    
    # input 0 for original audio, inputs 1+ for TTS files
    #filter_parts.append("[0:a]aformat=sample_fmts=fltp:sample_rates=16000:channel_layouts=mono[aud]")
    filter_parts.append("[0:a]aformat=sample_fmts=fltp:sample_rates=16000:channel_layouts=mono[aud]")

    # original audio with normalized volume
    filter_parts.append("[aud]volume=2[origvol]")
    
    # For each TTS, create a delayed and volume-adjusted version
    for idx, (_, delay_ms, _) in enumerate(tts_list, start=1):
        filter_parts.append(
            f"[{idx}:a]aformat=sample_fmts=fltp:sample_rates=16000:channel_layouts=mono,"
            f"adelay={delay_ms}|{delay_ms},volume=2[d{idx}]"
        )
    
    # mix all streams together
    #mix_inputs = "[aud]" + "".join(f"[d{i}]" for i in range(1, len(tts_list) + 1))
    #filter_parts.append(f"{mix_inputs}amix=inputs={len(tts_list) + 1}:duration=longest[out]")

    mix_inputs = "[origvol]" + "".join(f"[d{i}]" for i in range(1, len(tts_list) + 1))
    filter_parts.append(f"{mix_inputs}amix=inputs={len(tts_list) + 1}:dropout_transition=0:normalize=0[out]")
    
    # final filter complex
    filter_complex = ";".join(filter_parts)
    
    # input command string
    input_cmd = f'-i "{audio_path}"'
    for tts_file, _, _ in tts_list:
        input_cmd += f' -i "{tts_file}"'
    
    # mixed audio
    working_audio = os.path.join(working_dir, "final_full_audio.wav")
    mix_cmd = (
        f'ffmpeg -y {input_cmd} '
        f'-filter_complex "{filter_complex}" '
        f'-map "[out]" -ac 1 -ar 16000 "{working_audio}"'
    )
    print("\nMixing audio streams:")
    print(mix_cmd)
    run_command(mix_cmd)
    
    # mux with mp4
    mux_cmd = f'ffmpeg -y -i "{video_path}" -i "{working_audio}" -c:v copy -map 0:v:0 -map 1:a:0 -c:a aac "{output_video}"'
    print("\nMuxing with video:")
    print(mux_cmd)
    run_command(mux_cmd)
    
    print(f"\nFinal video saved to: {output_video}")

    shutil.rmtree(working_dir)
    print(f"Deleted working directory: {working_dir}")

if __name__ == "__main__":
    main()