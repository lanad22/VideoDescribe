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

def process_audio_segments(audio_path, silence_thresh=-35, min_silence_len=1000):
    """Process audio into segments with precise timestamp tracking."""
    audio = AudioSegment.from_file(audio_path)
    
    # First, detect silent ranges
    silent_ranges = silence.detect_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        seek_step=1
    )
    
    # Convert silent ranges to non-silent ranges
    chunks = []
    audio_length = len(audio)
    
    if not silent_ranges:
        # If no silence detected, process the whole audio
        chunks.append({
            'chunk': audio,
            'start': 0,
            'end': audio_length / 1000.0,
            'original_duration': audio_length / 1000.0
        })
        return chunks
    
    # Process ranges between silences
    last_end = 0
    for start_ms, end_ms in silent_ranges:
        if start_ms > last_end:
            # Extract the speech segment
            chunk = audio[last_end:start_ms]
            if len(chunk) > 100:  # Minimum chunk length of 100ms
                chunks.append({
                    'chunk': chunk,
                    'start': last_end / 1000.0,
                    'end': start_ms / 1000.0,
                    'original_duration': (start_ms - last_end) / 1000.0
                })
        last_end = end_ms
    
    # Don't forget the last chunk if it exists
    if last_end < audio_length:
        chunk = audio[last_end:audio_length]
        if len(chunk) > 100:
            chunks.append({
                'chunk': chunk,
                'start': last_end / 1000.0,
                'end': audio_length / 1000.0,
                'original_duration': (audio_length - last_end) / 1000.0
            })
    
    return chunks

def transcribe_chunk(model, chunk_data, idx, vad_threshold=0.5):
    """Transcribes audio chunk with precise timestamp tracking."""
    chunk_path = f"chunk_{idx}.wav"
    chunk_data['chunk'].export(chunk_path, format="wav")
    
    segments, _ = model.transcribe(
        chunk_path,
        vad_filter=True,
        vad_parameters={"threshold": vad_threshold},
        beam_size=8,
        temperature=0.1
    )

    os.remove(chunk_path)
    
    chunk_results = []
    chunk_start = chunk_data['start']
    chunk_end = chunk_data['end']
    
    for segment in segments:
        text = segment.text.strip()
        
        if not text or any(ending in text.lower() for ending in COMMON_ENDINGS):
            continue
        
        # Calculate timestamps based on the original audio position
        rel_start = segment.start
        rel_end = segment.end
        
        # Ensure the segment timing makes sense within the chunk
        if rel_start > chunk_data['original_duration']:
            continue
            
        seg_start = round(chunk_start + rel_start, 3)
        seg_end = round(chunk_start + rel_end, 3)
        
        # Validate segment boundaries
        if seg_end > chunk_end:
            seg_end = round(chunk_end, 3)
        
        if seg_end > seg_start:  # Only add valid segments
            chunk_results.append({
                'text': text,
                'start_time': seg_start,
                'end_time': seg_end
            })
    
    return chunk_results

def validate_and_adjust_segments(segments):
    """Validates and adjusts segment timestamps."""
    if not segments:
        return segments
    
    segments.sort(key=lambda x: x['start_time'])
    validated = []
    
    for i, current in enumerate(segments):
        if i > 0:
            prev = validated[-1]
            
            # Check for unrealistic gaps or overlaps
            gap = current['start_time'] - prev['end_time']
            
            # If segments overlap
            if gap < 0:
                # Adjust current segment to start after previous one
                current['start_time'] = round(prev['end_time'] + 0.1, 3)
                current['end_time'] = max(
                    round(current['start_time'] + 0.5, 3),
                    current['end_time']
                )
            
            # If gap is too large (but keep legitimate long pauses)
            elif gap > 10.0:  # Only adjust extremely large gaps
                # Investigate if this might be a timestamp error
                duration = current['end_time'] - current['start_time']
                current['start_time'] = round(prev['end_time'] + 1.0, 3)
                current['end_time'] = round(current['start_time'] + duration, 3)
        
        validated.append(current)
    
    return validated

def main(video_path):
    """Process audio and save transcripts as JSON."""
    output_dir = os.path.dirname(video_path)
    audio_path = extract_audio(video_path)
    
    print("\nInitializing Whisper model...")
    model = WhisperModel("large", device="cuda", compute_type="float16")
    
    print("\nProcessing audio segments...")
    chunks = process_audio_segments(
        audio_path,
        silence_thresh=-35,
        min_silence_len=1000  # Increased minimum silence length
    )
    
    all_segments = []
    total_chunks = len(chunks)
    
    for idx, chunk_data in enumerate(chunks):
        segments = transcribe_chunk(model, chunk_data, idx)
        all_segments.extend(segments)
        print(f"Chunk {idx+1}/{total_chunks} transcribed (Start: {chunk_data['start']:.2f}s)")
    
    # Validate and adjust timestamps
    all_segments = validate_and_adjust_segments(all_segments)
    
    # Save transcription
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