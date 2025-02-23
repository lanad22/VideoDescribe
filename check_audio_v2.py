import os
import json
import subprocess
from pydub import AudioSegment, silence
from faster_whisper import WhisperModel

# Common phrases to filter out
COMMON_ENDINGS = [
    "thank you for watching",
    "thanks for watching",
    "don't forget to subscribe",
    "like and subscribe",
    "hit the like button",
    "see you in the next video",
    'thank you'
]

def extract_audio(video_path):
    """Extracts audio from MP4 if WAV does not exist."""
    audio_path = video_path.replace(".mp4", ".wav")
    if not os.path.exists(audio_path):
        subprocess.run([
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1",
            audio_path
        ], check=True)
        print(f"Audio extracted to: {audio_path}")
    return audio_path

def detect_speech_chunks(audio_path, silence_thresh=-35, min_silence_len=2000):
    """Detect chunks of speech with significant silence between them."""
    audio = AudioSegment.from_file(audio_path)
    
    # First, detect silent ranges - using a longer minimum silence length
    silent_ranges = silence.detect_silence(
        audio,
        min_silence_len=min_silence_len,  # 2 seconds minimum silence
        silence_thresh=silence_thresh,
        seek_step=1
    )
    
    # Convert silent ranges to speech ranges
    chunks = []
    audio_length = len(audio)
    
    last_end = 0
    for start_ms, end_ms in silent_ranges:
        if start_ms > last_end:
            # Extract the speech segment
            chunk = audio[last_end:start_ms]
            if len(chunk) > 500:  # Minimum chunk length of 500ms
                chunks.append({
                    'chunk': chunk,
                    'start': last_end / 1000.0,
                    'end': start_ms / 1000.0,
                    'duration': (start_ms - last_end) / 1000.0
                })
        last_end = end_ms
    
    # Handle the last chunk
    if last_end < audio_length:
        chunk = audio[last_end:audio_length]
        if len(chunk) > 500:
            chunks.append({
                'chunk': chunk,
                'start': last_end / 1000.0,
                'end': audio_length / 1000.0,
                'duration': (audio_length - last_end) / 1000.0
            })
    
    return chunks

def process_chunk(model, chunk_data, idx):
    """Process individual chunk with VAD filtering."""
    chunk_path = f"chunk_{idx}.wav"
    chunk_data['chunk'].export(chunk_path, format="wav")
    
    segments, _ = model.transcribe(
        chunk_path,
        vad_filter=True,
        vad_parameters={
            "threshold": 0.4,
            "min_speech_duration_ms": 250,
            "min_silence_duration_ms": 100
        },
        beam_size=5,
        temperature=0.0
    )
    
    os.remove(chunk_path)
    
    chunk_results = []
    for segment in segments:
        text = segment.text.strip()
        
        if not text or any(ending in text.lower() for ending in COMMON_ENDINGS):
            continue
        
        # Calculate timestamps relative to original audio
        start_time = round(chunk_data['start'] + segment.start, 3)
        end_time = round(chunk_data['start'] + segment.end, 3)
        
        # Ensure end time doesn't exceed chunk boundary
        if end_time > chunk_data['end']:
            end_time = round(chunk_data['end'], 3)
            
        # Only add if duration makes sense
        if end_time > start_time:
            chunk_results.append({
                'text': text,
                'start_time': start_time,
                'end_time': end_time
            })
    
    return chunk_results

def main(video_path):
    """Process video and save transcript."""
    output_dir = os.path.dirname(video_path)
    audio_path = extract_audio(video_path)
    
    print("\nInitializing Whisper model...")
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    
    print("\nDetecting speech chunks...")
    chunks = detect_speech_chunks(
        audio_path,
        silence_thresh=-35,
        min_silence_len=2000  # 2 seconds minimum silence
    )
    
    print(f"\nFound {len(chunks)} distinct speech segments")
    
    all_segments = []
    for idx, chunk_data in enumerate(chunks):
        print(f"\nProcessing chunk {idx+1}/{len(chunks)} "
              f"(Start: {chunk_data['start']:.2f}s, Duration: {chunk_data['duration']:.2f}s)")
        
        segments = process_chunk(model, chunk_data, idx)
        all_segments.extend(segments)
    
    # Sort all segments by start time
    all_segments.sort(key=lambda x: x['start_time'])
    
    # Save results
    json_path = os.path.join(output_dir, 'transcript.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_segments, f, indent=2, ensure_ascii=False)
    
    print(f"\nTranscription saved to {json_path}")
    return json_path

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python script.py videos/video_id/video_id.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    main(video_path)