import os
import json
import difflib
import subprocess
import torch
import librosa
import numpy as np
import whisper_timestamped
import onnxruntime
onnxruntime.set_default_logger_severity(3)
os.environ["OMP_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")


from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor




WHISPER_MODEL = "large-v3"
WAV2VEC_MODEL = "facebook/wav2vec2-large-960h-lv60-self"

def match_captions(scene_start, scene_end, scene_duration, captions):
    """
    Given scene boundaries and a list of captions (each with 'start', 'duration', and 'text'),
    return a list of captions that substantially overlap with the scene.
    The returned caption start and end times are relative to the scene start.
    """
    scene_captions = []
    for caption in captions:
        cap_start = caption.get("start", 0)
        cap_duration = caption.get("duration", 0)
        cap_end = cap_start + cap_duration
        if cap_start < scene_end and cap_end > scene_start:
            overlap_start = max(cap_start, scene_start)
            overlap_end = min(cap_end, scene_end)
            if (overlap_end - overlap_start) >= (cap_duration * 0.5):
                scene_captions.append({
                    "text": caption.get("text", ""),
                    "start": max(cap_start - scene_start, 0),
                    "end": min(cap_end - scene_start, scene_duration)
                })
    return scene_captions

def extract_audio(scene_video_path, output_audio_path):
    """
    Extract audio from a video file (scene) using ffmpeg.
    Audio is extracted as a WAV file (mono, 16kHz).
    """
    command = [
        "ffmpeg", "-y",
        "-i", scene_video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "44100", #change from 16000
        "-ac", "1",
        output_audio_path
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Extracted audio: {output_audio_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio from {scene_video_path}: {e.stderr.decode()}")

def transcribe_whisper(wav_path, device="cuda"):
    """
    Transcribe the audio using Whisper via whisper_timestamped.
    Returns a list of transcript segments (each with text, start, and end times).
    """
    print(f"Transcribing with Whisper on audio: {wav_path}")
    try:
        model = whisper_timestamped.load_model(WHISPER_MODEL, device=device)
        result = whisper_timestamped.transcribe(
            model,
            wav_path,
            vad="silero:v3.1",
            beam_size=10,
            best_of=5,
            temperature=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
        )
        transcripts = []
        for segment in result["segments"]:
            transcripts.append({
                "text": segment["text"].strip(),
                "start": segment["start"],
                "end": segment["end"]
            })
        print(f"Whisper transcription complete: {len(transcripts)} segments")
        return transcripts
    except Exception as e:
        print(f"Error transcribing with Whisper: {str(e)}")
        return []

def transcribe_wav2vec(wav_path, device="cuda"):
    """
    Transcribe the given WAV file using Wav2Vec2.
    Returns a list of transcript segments.
    """
    print(f"Transcribing with Wav2Vec2 on audio: {wav_path}")
    try:
        processor = Wav2Vec2Processor.from_pretrained(WAV2VEC_MODEL)
        model = Wav2Vec2ForCTC.from_pretrained(WAV2VEC_MODEL).to(device)
    except Exception as e:
        print(f"Error loading Wav2Vec2 model: {str(e)}")
        return []
    try:
        speech_array, _ = librosa.load(wav_path, sr=16000)
        chunk_size = 16000 * 60  # 60 seconds per chunk
        transcripts = []
        for i in range(0, len(speech_array), chunk_size):
            chunk = speech_array[i : min(i + chunk_size, len(speech_array))]
            inputs = processor(chunk, sampling_rate=16000, return_tensors="pt").to(device)
            with torch.no_grad():
                logits = model(inputs.input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)[0].lower().strip()
            start_time = i / 16000.0
            end_time = min(i + chunk_size, len(speech_array)) / 16000.0
            transcripts.append({
                "text": transcription,
                "start": start_time,
                "end": end_time
            })
        print(f"Wav2Vec2 transcription complete: {len(transcripts)} segments")
        return transcripts
    except Exception as e:
        print(f"Error during Wav2Vec2 transcription: {str(e)}")
        return []

def verify_transcriptions(whisper_transcripts, wav2vec_transcripts):
    """
    Verify Whisper transcripts using Wav2Vec2 output.
    If Wav2Vec2 returns no segments at all, then discard all Whisper segments.
    Otherwise, for each Whisper segment, if there is no overlapping Wav2Vec2 segment,
    the Whisper segment is discarded.
    """
    for wav_seg in wav2vec_transcripts:
        if not wav_seg["text"]:
            return []
    verified = []
    for w_seg in whisper_transcripts:
        has_overlap = any(w_seg["start"] < w2_seg["end"] and w_seg["end"] > w2_seg["start"] 
                          for w2_seg in wav2vec_transcripts)
        if has_overlap:
            verified.append(w_seg)
    return verified

def should_discard_captions(global_transcript_text, global_caption_text, threshold=0.8):
    """
    Compare the global transcript and caption texts.
    Returns True if the similarity is above the threshold, indicating that captions should be discarded.
    """
    similarity = difflib.SequenceMatcher(None, global_transcript_text, global_caption_text).ratio()
    print(f"Global transcript vs captions similarity: {similarity:.2f}")
    return similarity >= threshold

def update_scene_transcripts(video_folder, device="cuda", verification_threshold=0.6, global_caption_threshold=0.8):
    video_id = os.path.basename(os.path.normpath(video_folder))
    scene_json_path = os.path.join(video_folder, f"{video_id}_scenes", "scene_info.json")
    
    if not os.path.exists(scene_json_path):
        print(f"Scene JSON file not found: {scene_json_path}")
        return

    with open(scene_json_path, "r") as f:
        scenes = json.load(f)

    # Load global captions if available (expects a JSON file named <video_id>.json in video_folder)
    captions = None
    captions_path = os.path.join(video_folder, f"{video_id}.json")
    if os.path.exists(captions_path):
        try:
            with open(captions_path, "r") as f:
                cap_data = json.load(f)
                captions = cap_data.get("captions", [])
                print(f"Loaded {len(captions)} captions from {captions_path}")
        except Exception as e:
            print(f"Error loading captions: {str(e)}")
    
    global_transcript_text = ""
    
    # Process each scene to update transcripts and build global transcript text
    for scene in scenes:
        scene_path = scene.get("scene_path", "")
        if not scene_path or not os.path.exists(scene_path):
            print(f"Scene video not found for scene {scene.get('scene_number')}, skipping transcription.")
            continue

        temp_audio_path = scene_path.replace(".mp4", ".wav")
        print(f"Extracting audio for scene {scene.get('scene_number')}")
        extract_audio(scene_path, temp_audio_path)

        # Transcribe with both models and verify
        whisper_trans = transcribe_whisper(temp_audio_path, device)
        print(f"WHISPER TRANS: {whisper_trans}")
        wav2vec_trans = transcribe_wav2vec(temp_audio_path, device)
        print(f"WAV2VEC2 TRANS: {wav2vec_trans}")
        verified_trans = verify_transcriptions(whisper_trans, wav2vec_trans)
        scene["transcript"] = verified_trans

        # Append the scene transcript text to the global transcript text
        scene_transcript_text = " ".join([seg["text"] for seg in verified_trans])
        global_transcript_text += scene_transcript_text + " "

        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
            print(f"Removed temporary audio file: {temp_audio_path}")

    # Global captions check: compare the entire transcript with the full captions text
    if captions and global_transcript_text.strip():
        global_caption_text = " ".join([cap["text"] for cap in captions])
        if should_discard_captions(global_transcript_text, global_caption_text, global_caption_threshold):
            print("Global captions are very similar to the transcript. Discarding captions for all scenes.")
            for scene in scenes:
                scene["captions"] = []
        else:
            # If captions are not similar, match captions to each scene based on scene boundaries
            for scene in scenes:
                scene_start = scene.get("start", 0)
                scene_end = scene.get("end", 0)
                if scene_end > scene_start:
                    scene_duration = scene_end - scene_start
                    scene_captions = match_captions(scene_start, scene_end, scene_duration, captions)
                    scene["captions"] = scene_captions
                    print(f"Scene {scene.get('scene_number')}: {len(scene_captions)} captions matched.")
                else:
                    scene["captions"] = []
    else:
        # No captions available or transcript text is empty
        for scene in scenes:
            scene["captions"] = []
    
    with open(scene_json_path, "w") as out_f:
        json.dump(scenes, out_f, indent=2)
    print(f"Updated scene JSON with transcripts saved to: {scene_json_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Transcribe scene audio using Whisper and Wav2Vec2, verify transcripts, and update scene JSON with optional captions."
    )
    parser.add_argument("video_folder", type=str,
                        help="Path to the video folder (e.g., videos/video_id). The scene_info.json file is expected at videos/video_id/video_id_scenes_new/scene_info.json")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use for transcription (default: cuda)")
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="Similarity threshold for transcription verification (default: 0.6)")
    
    args = parser.parse_args()
    update_scene_transcripts(args.video_folder, args.device, args.threshold)
