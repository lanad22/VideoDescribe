README

1) youtube_downloader.py
   - Usage:
       python youtube_downloader.py "YOUTUBE_URL"
   - Downloads the video (and any available captions) into a folder: videos/<VIDEO_ID>/<VIDEO_ID>.mp4 and <VIDEO_ID>_captions.json.

2) scene_detector.py
   - Usage:
       python scene_detector.py <video_path> [--captions <captions_path>]
   - Automatically detects scene boundaries and splits the video into multiple .mp4 files. Saves scene metadata (including optional captions) to scene_info.json.

Typical Workflow:
1. Download a video:
   python youtube_downloader.py "https://youtube.com/watch?v=XYZ"
2. Detect scenes:
   python scene_detector.py videos/XYZ/XYZ.mp4 --captions videos/XYZ/XYZ_captions.json
3. Output:
   videos/XYZ_scenes/scene_001.mp4, scene_002.mp4, etc., plus scene_info.json.

