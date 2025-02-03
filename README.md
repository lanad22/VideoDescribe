INSTALLATION
------------

1. Create and activate conda environment:  

   conda create -n video_env python=3.9  
   conda activate video_env  

2. Install FFmpeg:  

   conda install ffmpeg -c conda-forge  

3. Install required packages:  

   pip install -r requirements.txt  


HOW TO USE
----------

1) youtube_downloader.py  

   • Description: Downloads a YouTube video (and captions, if available).  

   • Usage:  

       python youtube_downloader.py "YOUTUBE_URL"  

   • Output:  

       videos/<VIDEO_ID>/<VIDEO_ID>.mp4  

       videos/<VIDEO_ID>/<VIDEO_ID>_captions.json (if captions exist)  


2) video_processor.py  

   • Description: Detects scenes, transcribes speech, and prepares videos for Qwen.  

   • Usage:  

       python video_processor.py <video_path> \  
           [--whisper-model <model>] \  
           [--captions <captions_path>] \  
           [--min-duration <seconds>]  

   • Arguments:  

       --whisper-model: Whisper model size (tiny/base/small/medium/large, default: base)  
       --captions: Path to captions file (optional)  
       --min-duration: Minimum scene duration in seconds (default: 5.0)  

   • Output:  

       <video_stem>_scenes/scene_001.mp4      # Qwen-compatible video  
       <video_stem>_scenes/scene_001.txt      # Audio transcript (if speech detected)  
       <video_stem>_scenes/scene_002.mp4  
       <video_stem>_scenes/scene_002.txt  
       <video_stem>_scenes/scene_info.json    # Scene metadata  


EXAMPLE
-------

1. Download the video:  

   python youtube_downloader.py "https://www.youtube.com/watch?v=EXAMPLE"  

2. Process video:  

   python video_processor.py videos/EXAMPLE/EXAMPLE.mp4 \  
      --whisper-model base \  
      --captions videos/EXAMPLE/EXAMPLE_captions.json