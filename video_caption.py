import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import subprocess
import hashlib
import json
import numpy as np
import torch
import torchvision
import transformers
import decord
from decord import VideoReader, cpu
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, GenerationConfig
from qwen_vl_utils import process_vision_info
from gtts import gTTS
import tempfile


def get_tts_duration(text):
    """Generate TTS and return its duration"""
    if not text or text.isspace():
        return 0.0
        
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=True) as temp_file:
        tts = gTTS(text=text, lang='en')
        tts.save(temp_file.name)
        cmd = f'ffprobe -v error -select_streams a:0 -show_entries format=duration -of csv="p=0" "{temp_file.name}"'
        duration = float(subprocess.check_output(cmd, shell=True).decode().strip())
        return duration

def find_largest_gap(transcripts, scene_duration):
    """Find the largest gap between transcripts"""
    if not transcripts:
        return scene_duration, 0.0

    segments = sorted(transcripts, key=lambda x: x["start"])
    largest_gap = 0
    gap_start = 0

    # Check gap at start
    if segments[0]["start"] > largest_gap:
        largest_gap = segments[0]["start"]
        gap_start = 0

    # Check gaps between segments
    for i in range(len(segments) - 1):
        gap = segments[i+1]["start"] - segments[i]["end"]
        if gap > largest_gap:
            largest_gap = gap
            gap_start = segments[i]["end"]

    # Check gap at end
    end_gap = scene_duration - segments[-1]["end"]
    if end_gap > largest_gap:
        largest_gap = end_gap
        gap_start = segments[-1]["end"]

    return largest_gap, gap_start

def get_video_frames(scene_path, target_fps=2):
    video_dir = os.path.dirname(scene_path)
    cache_dir = os.path.join(video_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    video_hash = hashlib.md5(scene_path.encode('utf-8')).hexdigest()
    frames_cache_file = os.path.join(cache_dir, f'{video_hash}_frames.npy')
    
    if os.path.exists(frames_cache_file):
        frames = np.load(frames_cache_file)
        return frames
    
    vr = VideoReader(scene_path, ctx=cpu(0))
    total_frames = len(vr)
    video_fps = vr.get_avg_fps()
    duration = total_frames / video_fps  
    
    if total_frames == 0:
        print(f"Warning: No frames found in video {scene_path}")
        return np.array([])
    
    # Calculate frame indices to sample
    step = video_fps / target_fps  # how many original frames to skip
    num_samples = int(duration * target_fps)  # total number of frames will be extracted
    frame_indices = [int(i * step) for i in range(num_samples)]
    
    # Always include the first and last frame
    if 0 not in frame_indices:
        frame_indices.insert(0, 0)
    if total_frames - 1 not in frame_indices:
        frame_indices.append(total_frames - 1)
    
    frame_indices = sorted(list(set(frame_indices)))
    frame_indices = [idx for idx in frame_indices if idx < total_frames]
    
    print(f"Extracting frames from {scene_path}")
    print(f"Video duration: {duration:.2f}s")
    print(f"Original FPS: {video_fps:.2f}")
    print(f"Process FPS: {target_fps}")
    print(f"Total frames extracted: {len(frame_indices)}")
    
    try:
        # Extract the frames
        frames = vr.get_batch(frame_indices).asnumpy()
        np.save(frames_cache_file, frames)
        return frames
        
    except Exception as e:
        print(f"Error extracting frames from {scene_path}: {str(e)}")
        return np.array([])
'''
def get_video_frames(scene_path):
    video_dir = os.path.dirname(scene_path)
    cache_dir = os.path.join(video_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    video_hash = hashlib.md5(scene_path.encode('utf-8')).hexdigest()
    frames_cache_file = os.path.join(cache_dir, f'{video_hash}_frames.npy')
    
    if os.path.exists(frames_cache_file):
        frames = np.load(frames_cache_file)
        return frames
    
    cmd = f'ffprobe -select_streams v -show_frames -show_entries frame=pict_type,pkt_pts_time -of csv "{scene_path}"'
    result = subprocess.check_output(cmd, shell=True).decode()
    
    iframe_indices = []
    frame_idx = 0
    for line in result.split('\n'):
        if 'I' in line:
            iframe_indices.append(frame_idx)
        frame_idx += 1
    
    iframe_indices = np.array(iframe_indices)
    print(f"Number of I-frames for {scene_path}: {len(iframe_indices)}")
    
    vr = VideoReader(scene_path, ctx=cpu(0))
    frames = vr.get_batch(iframe_indices).asnumpy()
    
    np.save(frames_cache_file, frames)
    return frames
'''
def generate_scene_caption(scene_data, available_gap, previous_caption=None, max_new_tokens=2048,
                         total_pixels=20480 * 28 * 28, min_pixels=16 * 28 * 28):

    '''
    if available_gap < 1:  # Skip if gap is too small
        return {"text": "", "type": "none"}'''
        
    scene_path = scene_data.get('scene_path', "unknown_scene_path")

    # Build the context from previous captions, provided captions, and transcripts
    context_parts = []
    if previous_caption:
        context_parts.append(
            f"Background context (for reference only, do not repeat):\n{previous_caption}\n\n"
            "Focus on new observations, actions, and details from the current scene.\n"
        )
    
    if scene_data and scene_data.get("captions"):
        captions_text = "\n".join([f"- {caption}" for caption in scene_data["captions"]])
        context_parts.append(f"Scene {scene_data.get('scene_number', '')} captions:\n{captions_text}\n")
    
    if scene_data and scene_data.get("transcripts"):
        audio_text = "\n".join([f"- {transcript['text']}" for transcript in scene_data["transcripts"]])
        context_parts.append(
            f"Audio transcript for Scene {scene_data.get('scene_number', '')}:\n{audio_text}\n"
            "Do not repeat this information in the description.\n"
        )
    

    context = "\n".join(context_parts)
    max_words = int(available_gap * 1.5)
    
    prompt = (
    f"{context}\n\n"
    "Analyze the scene by integrating visual, audio, and on-screen text, along with previous context.\n"
    "If on-screen text is present, state it explicitly.\n"
    "Provide a contextually rich description using the fewest syllables possible.\n"
    f"Ensure the description fits within {available_gap:.1f} seconds when spoken, targeting no more than {max_words} words.\n"
    )
    
  
    messages = [
                {"role": "system", "content": "You are an expert audio describer."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"video": scene_path, "total_pixels": total_pixels, "min_pixels": min_pixels},
                ]},
        ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True)
    fps_inputs = video_kwargs['fps']
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, fps=fps_inputs,
                            padding=True, return_tensors="pt")

    
    inputs = inputs.to('cuda')
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9
    )
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    
    try:
        duration = get_tts_duration(caption)
        print(f"Caption duration = {duration:.1f}s (target: {available_gap:.1f}s)")
        
        # Classify caption based on duration
        if duration <= (available_gap + 1.5):
            return {"text": caption, "type": "inline"}
        else:
            return {"text": caption, "type": "extended"}
            
    except Exception as e:
        print(f"Error checking duration: {str(e)}")
        return {"text": caption, "type": "extended"}  # Default to extended if duration check fails

def process_all_scenes(video_folder):
    video_id = os.path.basename(video_folder)
    video_metadata_path = os.path.join(video_folder, f"{video_id}.json")
    scenes_folder = os.path.join(video_folder, f"{video_id}_scenes")
    scenes_json_path = os.path.join(scenes_folder, "scene_info.json")

    if not os.path.exists(video_metadata_path):
        print(f"Error: {video_metadata_path} not found. Unable to retrieve title and description.")
        return

    with open(video_metadata_path, "r") as f:
        video_metadata = json.load(f)

    video_title = video_metadata.get("title", "Unknown Title")
    video_description = video_metadata.get("description", "")
    previous_caption = f"Video Title: {video_title}\n{video_description}\n"
    #print(previous_caption)
    
    if not os.path.exists(scenes_json_path):
        print(f"Error: scene_info.json not found in {scenes_folder}")
        return

    with open(scenes_json_path, "r") as f:
        scene_list = json.load(f)

    print(f"Processing {len(scene_list)} scenes in {scenes_folder}...")


    for idx, scene_data in enumerate(scene_list, start=1):
        print(f"\nProcessing Scene {idx}: {scene_data.get('scene_number', 'Unknown')}")
        
        scene_duration = scene_data.get("duration")
        if scene_duration is None:
            scene_end = scene_data.get("end_time", 0)
            scene_start = scene_data.get("start_time", 0)
            scene_duration = scene_end - scene_start
        
        largest_gap, gap_start = find_largest_gap(scene_data.get("transcripts", []), scene_duration)
        print(f"Largest available gap: {largest_gap:.2f} seconds")

        '''
        if largest_gap < 1:
            print("Gap too small (<1s), skipping caption generation")
            scene_data["description"] = ""
            continue
        '''
          
        caption_result = generate_scene_caption( 
            scene_data,
            largest_gap,
            previous_caption
        )
        
        if caption_result["text"]:
            try:
                tts_duration = get_tts_duration(caption_result["text"])
                print(f"Final caption duration: {tts_duration:.2f}s")
            except Exception as e:
                print(f"Error checking final TTS duration: {str(e)}")
            
            print(f"Generated caption ({caption_result['type']}):")
            print(caption_result["text"])
            scene_data["description"] = caption_result["text"]
            scene_data["description_type"] = caption_result["type"]
            previous_caption += caption_result["text"] + "\n"
        else:
            print("No caption generated for this scene")
            scene_data["description"] = ""
            scene_data["description_type"] = "none"

    with open(scenes_json_path, "w") as f:
        json.dump(scene_list, f, indent=4)

    print(f"\nScene descriptions updated in: {scenes_json_path}")
    return scene_list

def main():
    parser = argparse.ArgumentParser(
        description="Generate scene captions from video scenes and update scene_info.json with descriptions."
    )
    parser.add_argument("video_folder", type=str,
                        help="Folder containing files associated video files.")
    args = parser.parse_args()

    global model, processor
    model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
    print("Loading model and processor...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path)
    print("Model and processor loaded.")

    process_all_scenes(args.video_folder)

if __name__ == "__main__":
    main()