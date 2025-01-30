import os
import time
import json
import argparse
from pathlib import Path
from youtube_transcript_api import YouTubeTranscriptApi

def extract_video_id(url: str) -> str:
    """Extract video ID from YouTube URL"""
    if 'v=' in url:
        return url.split('v=')[1].split('&')[0]
    return url.split('/')[-1]

def download_with_captions(url: str):
    """Download YouTube video and captions into a folder named after the video ID"""
    
    # Get video ID
    video_id = extract_video_id(url)

    # Create a folder with the video ID
    output_dir = os.path.join(os.getcwd(), "videos", video_id)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Download captions
    captions_path = os.path.join(output_dir, f"{video_id}_captions.json")
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        with open(captions_path, 'w') as f:
            json.dump(transcript, f, indent=2)
        print(f"Captions downloaded to: {captions_path}")
    except Exception as e:
        print(f"Could not download captions: {str(e)}")
        captions_path = None

    # Get initial file list
    initial_files = set(os.listdir(output_dir))

    # Download video
    video_path = os.path.join(output_dir, f"{video_id}.mp4")
    command = f'yt-dlp -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4" -o "{video_path}" {url}'
    print("\nDownloading video...")
    os.system(command)

    time.sleep(2)

    if os.path.exists(video_path):
        print(f"Video downloaded to: {video_path}")
        print(f"File size: {os.path.getsize(video_path) / (1024*1024):.2f} MB")
    else:
        print("No MP4 file found")
        video_path = None

    return video_path, captions_path

class CaptionProcessor:
    def __init__(self, captions_path: str):
        """Load and process video captions"""
        with open(captions_path, 'r') as f:
            self.captions = json.load(f)

    def get_caption_at_time(self, timestamp: float) -> str:
        """Get caption at specific timestamp"""
        for caption in self.captions:
            start = caption['start']
            end = start + caption['duration']
            if start <= timestamp <= end:
                return caption['text']
        return ""

    def get_captions_for_segment(self, start_time: float, end_time: float) -> str:
        """Get all captions for a time segment"""
        relevant_captions = []
        for caption in self.captions:
            caption_start = caption['start']
            caption_end = caption_start + caption['duration']

            # Check if caption overlaps with segment
            if (caption_start <= end_time and caption_end >= start_time):
                relevant_captions.append(caption['text'])

        return " ".join(relevant_captions)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download YouTube videos with captions.")
    parser.add_argument("url", help="YouTube video URL")
    args = parser.parse_args()

    url = args.url
    video_path, captions_path = download_with_captions(url)

    if captions_path:
        caption_processor = CaptionProcessor(captions_path)
        early_captions = caption_processor.get_captions_for_segment(0, 10)
        print("\nCaptions from first 10 seconds:")
        print(early_captions)
