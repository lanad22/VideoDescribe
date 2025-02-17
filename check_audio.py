import os
import json
import subprocess
from pydub import AudioSegment
from faster_whisper import WhisperModel

# Common phrases to filter out
COMMON_ENDINGS = [
    "thank you for watching",
    "thanks for watching",
    "don't forget to subscribe",
    "like and subscribe",
    "hit the like button",
    "see you in the next video"
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

def split_audio(audio_path, chunk_length_ms=30000, overlap_ms=8000):
    """Splits audio into overlapping chunks."""
    audio = AudioSegment.from_file(audio_path)
    step = chunk_length_ms - overlap_ms
    chunks = []
    
    for i in range(0, len(audio), step):
        chunk = audio[i:i+chunk_length_ms]
        start_time = i / 1000.0  # Convert to seconds
        chunks.append((chunk, start_time))
    return chunks

def transcribe_chunk(model, chunk, chunk_start_time, idx, vad_threshold=0.4):
    """Transcribes audio chunk with adjustable VAD and beam search."""
    chunk_path = f"chunk_{idx}.wav"
    chunk.export(chunk_path, format="wav")
    
    segments, _ = model.transcribe(
        chunk_path,
        vad_filter=True,
        vad_parameters={"threshold": vad_threshold},
        beam_size=8,
        temperature=0.1
    )

    os.remove(chunk_path)
    
    chunk_results = []
    for segment in segments:
        text = segment.text.strip()
        
        # Skip empty or common ending phrases
        if not text or any(ending in text.lower() for ending in COMMON_ENDINGS):
            continue
            
        # Calculate absolute timestamps
        abs_start = chunk_start_time + segment.start
        abs_end = chunk_start_time + segment.end
        
        chunk_results.append({
            'text': text,
            'start_time': abs_start,
            'end_time': abs_end
        })
    
    return chunk_results

def main(video_path):
    """Process audio and save transcripts as JSON."""
    # Get video_id from path (assuming format: videos/video_id/video_id.mp4)
    video_id = os.path.basename(os.path.dirname(video_path))
    output_dir = os.path.dirname(video_path)  # Use the same directory as video
    
    audio_path = extract_audio(video_path)
    
    print("\nInitializing Whisper model...")
    model = WhisperModel("large", device="cuda", compute_type="float16")
    
    all_segments = []
    print("\nProcessing audio in chunks...")
    for idx, (chunk, start_time) in enumerate(split_audio(audio_path, overlap_ms=8000)):
        segments = transcribe_chunk(model, chunk, start_time, idx)
        all_segments.extend(segments)
        print(f"Chunk {idx+1} transcribed")
    
    # Sort segments by start time
    all_segments.sort(key=lambda x: x['start_time'])
    
    # Save JSON format with timestamps
    json_path = os.path.join(output_dir, 'transcript.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_segments, f, indent=2, ensure_ascii=False)
    
    print(f"\nTranscription saved to {json_path}")
    return json_path

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python check_audio.py videos/video_id/video_id.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    main(video_path)