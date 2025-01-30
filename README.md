INSTALLATION
------------
1. Ensure you have Python 3.7+ installed.
2. (Optional) Create/activate a virtual environment.
3. Install required packages:
   pip install -r requirements.txt


HOW TO USE
----------
1) youtube_downloader.py \n\n
   • Description: Downloads a YouTube video (and captions, if available) \n\n
   • Usage: python youtube_downloader.py "YOUTUBE_URL" \n\n
   • Output: \n\n
       videos/<VIDEO_ID>/<VIDEO_ID>.mp4 \n\n
       videos/<VIDEO_ID>/<VIDEO_ID>_captions.json (if captions exist)

2) scene_detector.py
   • Description: Detects scene boundaries and splits the video into multiple .mp4 files.
   • Usage:
       python scene_detector.py <video_path> [--captions <captions_path>]
   • Output:
       <video_stem>_scenes/scene_001.mp4, scene_002.mp4, ...
       <video_stem>_scenes/scene_info.json

EXAMPLE
-------
   1. Download the video:
      python youtube_downloader.py "https://www.youtube.com/watch?v=EXAMPLE"
   2. Detect scenes:
      python scene_detector.py videos/EXAMPLE/EXAMPLE.mp4 \
         --captions videos/EXAMPLE/EXAMPLE_captions.json

