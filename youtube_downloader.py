import os
import json
import argparse
import subprocess
from pathlib import Path
from youtube_transcript_api import YouTubeTranscriptApi

def extract_video_id(url: str) -> str:
    """Extract video ID from YouTube URL"""
    if 'v=' in url:
        return url.split('v=')[1].split('&')[0]
    return url.split('/')[-1]

def get_video_metadata(url: str) -> dict:
    """Fetch video title and description using yt-dlp"""
    command = ["yt-dlp", "--get-title", "--get-description", url]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        metadata = result.stdout.strip().split("\n", 1)
        title = metadata[0] if metadata else "Unknown Title"
        description = metadata[1] if len(metadata) > 1 else ""  # Keep empty if no description
        return {"title": title, "description": description}
    except subprocess.CalledProcessError as e:
        print(f"Error fetching metadata: {e}")
        return {"title": "Unknown Title", "description": ""}

def download_with_captions(url: str):
    """Download YouTube video, metadata, and captions into a folder named after the video ID"""
    
    # Get video ID
    video_id = extract_video_id(url)

    # Create a folder with the video ID
    output_dir = os.path.join(os.getcwd(), "videos", video_id)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Fetch video metadata (title & description)
    metadata = get_video_metadata(url)

    # Prepare JSON data
    captions_path = os.path.join(output_dir, f"{video_id}.json")
    captions_data = {"title": metadata["title"], "description": metadata["description"], "captions": []}

    # Try downloading captions
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        captions_data["captions"] = transcript
        print("Captions successfully downloaded.")
    except Exception as e:
        print(f"Captions not available: {str(e)}")

    # Save JSON file with title, description, and (if available) captions
    with open(captions_path, 'w', encoding='utf-8') as f:
        json.dump(captions_data, f, indent=2, ensure_ascii=False)
    
    print(f"Metadata saved to: {captions_path}")

    # Download video
    video_path = os.path.join(output_dir, f"{video_id}.mp4")
    command = f'yt-dlp -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4" -o "{video_path}" {url}'
    print("\nDownloading video...")
    os.system(command)

    if os.path.exists(video_path):
        print(f"Video downloaded to: {video_path}")
        print(f"File size: {os.path.getsize(video_path) / (1024*1024):.2f} MB")
    else:
        print("No MP4 file found")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download YouTube videos with captions and metadata.")
    parser.add_argument("url", help="YouTube video URL")
    args = parser.parse_args()

    download_with_captions(args.url)
