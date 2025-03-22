import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import re
import ast
import argparse
import subprocess
import torch
import gc
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def convert_for_qwen(input_path):
    base_path, ext = os.path.splitext(input_path)
    output_path = f"{base_path}_qwen{ext}"
    
    command = [
        "ffmpeg", "-y",
        "-loglevel", "quiet",
        "-i", input_path,
        "-an",
        "-c:v", "libx264",
        "-vf", "scale='min(1280,iw)':'-2'",
        "-pix_fmt", "yuv420p",
        output_path
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path

def extract_and_parse_json(response):
    try:
        # Remove markdown code block markers if present
        response = re.sub(r'```json|```', '', response)
        json_pattern = r'\[\s*{.*}\s*\]'
        json_match = re.search(json_pattern, response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0).strip()
            try:
                parsed_data = json.loads(json_str)
                return parsed_data
            except json.JSONDecodeError:
                try:
                    parsed_data = ast.literal_eval(json_str)
                    return parsed_data
                except Exception:
                    return []
        else:
            return []
    except Exception:
        return []

def extract_structured_response_and_json(response):
    # Try to find JSON part first using the existing function
    json_data = extract_and_parse_json(response)
    
    # Extract guidelines understanding (YES response)
    guidelines_match = re.search(r'YES[^\n]*', response)
    guidelines_response = guidelines_match.group(0) if guidelines_match else "No explicit confirmation"
    
    # Extract context understanding (typically starts after "Based on this context")
    context_pattern = r'Based on this context.*?(?=Now, watch the video|Finally, provide|$)'
    context_match = re.search(context_pattern, response, re.DOTALL)
    context_understanding = context_match.group(0).strip() if context_match else "No explicit context analysis"
    
    # Extract visual elements identified (typically starts after "watch the video scene")
    visual_pattern = r'Now, watch the video.*?(?=Finally, provide|$)'
    visual_match = re.search(visual_pattern, response, re.DOTALL)
    visual_analysis = visual_match.group(0).strip() if visual_match else "No explicit visual analysis"
    
    # Print the model's structured understanding
    print("\n--- MODEL'S STRUCTURED UNDERSTANDING ---")
    print(f"GUIDELINES CONFIRMATION: {guidelines_response}")
    print(f"\nCONTEXT UNDERSTANDING: {context_understanding}")
    print(f"\nVISUAL ANALYSIS: {visual_analysis}")
    print("--- END OF STRUCTURED UNDERSTANDING ---\n")
    
    return json_data

def prepare_context(scene_data, previous_description=None):
    """Prepare context information from transcript and previous description."""
    context_parts = []
    
    # Add transcript information
    if scene_data.get("transcript"):
        transcripts_info = ""
        for transcript in scene_data.get("transcript"):
            transcripts_info += f"{transcript['text']}\n"
        context_parts.append(
            f"TRANSCRIPT:\n{transcripts_info}\n"
        )
    
    # Add captions information
    if scene_data.get("captions"):
        captions_info = ""
        for caption in scene_data.get("captions"):
            captions_info += f"{caption['text']}\n"
        context_parts.append(f"CAPTIONS:\n{captions_info}\n")
    
    # Add previous description if available
    if previous_description:
        context_parts.append(
            f"PREVIOUS SCENE CONTEXT:\n{previous_description}\n"
        )
    
    return "\n\n".join(context_parts)

def process_scene(scene_data, scene_path, model, processor, previous_description=None, system_message=None):
    # Prepare context
    context = prepare_context(scene_data, previous_description)
    scene_duration = scene_data.get("duration")
    
    # Create a more concise scene prompt that focuses on context review
    prompt = f"""
        SCENE DURATION: {scene_duration:.2f} seconds

        CONTEXT:
        {context}

        Based on this context, first identify what characters, locations, and important elements are present. 
        List the specific character names you'll use in your descriptions.

        Then, create a JSON array of events for this scene, including:
        1. On-screen text events (type: "text") with their exact text and start times. 
        IMPORTANT: DO NOT INCLUDE TRANSCRIPT OR CAPTIONS TEXT.
        2. Visual description events (type: "visual") using proper character names from the context and start times.

        Remember: ALWAYS use specific character names from the context, never generic terms.
    """
    
    print(f"SCENE PROMPT: {prompt}")
    
    if not system_message:
        system_message = "You are a professional audio describer."
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"video": scene_path, "total_pixels": 20480 * 28 * 28, "min_pixels": 16 * 28 * 28}
        ]}
    ]
    
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True)
    fps_inputs = video_kwargs['fps']
    
    inputs = processor(text=[text_prompt], images=image_inputs, videos=video_inputs,
                      fps=fps_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to('cuda')
    
    output_ids = model.generate(**inputs, max_new_tokens=2048)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    
    response = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    print(f'RAW RESPONSE: {response}')
    
    # Extract context understanding
    context_pattern = r'Based on this context.*?(?=\[|$)'
    context_match = re.search(context_pattern, response, re.DOTALL)
    context_understanding = context_match.group(0).strip() if context_match else "No explicit context analysis"
    
    print("\n--- MODEL'S CONTEXT UNDERSTANDING ---")
    print(context_understanding)
    print("--- END OF CONTEXT UNDERSTANDING ---\n")
    
    # Extract the JSON events
    events = extract_and_parse_json(response)
    
    # Convert any "start_time" strings to floats
    if not isinstance(events, list):
        print(f"Warning: Could not extract valid JSON from response. Using empty list.")
        events = []
    
    for event in events:
        if "start_time" in event and isinstance(event["start_time"], str):
            try:
                event["start_time"] = float(event["start_time"])
            except ValueError:
                event["start_time"] = 0.0
    
    # Deduplicate text events (if same text appears multiple times)
    unique_texts = {}
    deduplicated_events = []
    
    for event in events:
        if event.get("type") == "text":
            text_content = event.get("text", "").strip()
            if text_content and text_content not in unique_texts:
                unique_texts[text_content] = True
                deduplicated_events.append(event)
        else:
            deduplicated_events.append(event)
    
    if len(events) != len(deduplicated_events):
        print(f"Deduplicated text events: {len(events)} → {len(deduplicated_events)}")
    
    deduplicated_events.sort(key=lambda e: e.get("start_time", 0))
    return deduplicated_events


def process_all_scenes(video_folder, model, processor, skip_guidelines=False):
    """Process all scenes in the provided video folder with initial model training on guidelines."""
    video_id = os.path.basename(os.path.normpath(video_folder))
    video_metadata_path = os.path.join(video_folder, f"{video_id}.json")
    scenes_folder = os.path.join(video_folder, f"{video_id}_scenes")
    scenes_json_path = os.path.join(scenes_folder, "scene_info.json")
    
    if not os.path.exists(video_metadata_path):
        print(f"Error: {video_metadata_path} not found. Unable to retrieve title and description.")
        return
    
    # Load video metadata for context
    with open(video_metadata_path, "r") as f:
        video_metadata = json.load(f)
    
    video_title = video_metadata.get("title", "Unknown Title")
    video_description = video_metadata.get("description", "")
    previous_description = f"Video Title: {video_title}\n{video_description}"
    
    if not os.path.exists(scenes_json_path):
        print(f"Error: scene_info.json not found in {scenes_folder}")
        return
    
    with open(scenes_json_path, "r") as f:
        scene_list = json.load(f)
    
    print(f"Processing {len(scene_list)} scenes in {scenes_folder}...")
    
    # Define guidelines
    guidelines = """
    AUDIO DESCRIPTION GUIDELINES:
    - Describe what you see in a concise, factual manner.
    - Always read on-screen text exactly as it appears.
    - Be factual, objective, and precise in your descriptions.
    - Use proper terminology and names from the context when possible.
    - Match the tone and mood of the video.
    - Do not describe what can be inferred from the audio.
    - Do not over-describe - less is more.
    - Do not interpret or editorialize about what you see.
    - Do not give away surprises before they happen.
    - DO NOT mention or describe any camera movements or transitions.
    - Avoid phrases such as "camera pans," "camera tilts," "camera shifts," "camera zooms," or any language suggesting a change in camera angle or movement.
    
    IMPORTANT CHARACTER IDENTIFICATION:
    - When you recognize a character from the context, ALWAYS use their specific name.
    - Before each scene, carefully review context to identify all named characters.
    - Use the most specific identification possible based on the context information.
    """
    
    # First, have the model understand the guidelines by sending a one-time prompt
    if not skip_guidelines:
        print("\n----- ESTABLISHING GUIDELINES WITH MODEL -----")
        guidelines_prompt = f"""
        You are a professional audio describer following these guidelines:
        
        {guidelines}
        
        Do you understand these guidelines? Respond with "YES" and a brief confirmation.
        """
        
        guidelines_messages = [
            {"role": "system", "content": "You are a professional audio describer."},
            {"role": "user", "content": guidelines_prompt}
        ]
        
        guidelines_text_prompt = processor.apply_chat_template(guidelines_messages, tokenize=False, add_generation_prompt=True)
        guidelines_inputs = processor(text=[guidelines_text_prompt], return_tensors="pt")
        guidelines_inputs = guidelines_inputs.to('cuda')
        
        guidelines_output_ids = model.generate(**guidelines_inputs, max_new_tokens=512)
        guidelines_generated_ids = [guidelines_output_ids[len(input_ids):] for input_ids, guidelines_output_ids in zip(guidelines_inputs.input_ids, guidelines_output_ids)]
        
        guidelines_response = processor.batch_decode(guidelines_generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        print(f"MODEL RESPONSE: {guidelines_response}")
        print("----- GUIDELINES ESTABLISHED -----\n")
    
    # Create a system message that includes the guidelines reference
    system_message = "You are a professional audio describer following strict guidelines on using proper character names and avoiding generic descriptions."
    
    # Process each scene, building context from previous scenes
    for idx, scene_data in enumerate(scene_list, start=1):
        print(f"\nProcessing Scene {idx}/{len(scene_list)}: {scene_data.get('scene_number', 'Unknown')}")
        scene_path = scene_data.get('scene_path', '')
        if not os.path.exists(scene_path):
            print(f"Scene file not found: {scene_path}")
            continue
        
        try:
            print("Converting video for Qwen format...")
            converted_scene_path = convert_for_qwen(scene_path)
            
            try:
                # Process scene with our streamlined approach
                scene_events = process_scene(
                    scene_data, 
                    converted_scene_path, 
                    model, 
                    processor, 
                    previous_description=previous_description,
                    system_message=system_message
                )
                
                scene_data['audio_clips'] = scene_events
                
                print(f"Generated {len(scene_events)} events:")
                for event in scene_events:
                    st = event.get('start_time', 'N/A')
                    print(f"[{st}] ({event.get('type', 'unknown')}) {event.get('text', '')}")
                
                # Update previous_description with the current scene description for context in next scene
                visual_descriptions = [event.get('text', '') for event in scene_events if event.get('type') == 'visual']
                if visual_descriptions:
                    # Include previous description to maintain context across scenes
                    previous_description = f"{previous_description}\n\nMost recent scene: {visual_descriptions[0]}"
                
            except Exception as e:
                print(f"Error processing scene {idx}: {str(e)}")
                scene_data['audio_clips'] = []
            
            if converted_scene_path != scene_path and os.path.exists(converted_scene_path):
                os.remove(converted_scene_path)
                print(f"Removed temporary Qwen format file: {converted_scene_path}")
        
        except Exception as e:
            print(f"Error converting video: {str(e)}")
            scene_data['audio_clips'] = []
    
    with open(scenes_json_path, "w") as f:
        json.dump(scene_list, f, indent=4)
    
    print(f"\nScene descriptions updated in: {scenes_json_path}")
    return scene_list

    
def main():
    parser = argparse.ArgumentParser(
        description="Generate audio descriptions for video scenes."
    )
    parser.add_argument("video_folder", type=str,
                        help="Folder containing video files and metadata.")
    parser.add_argument("--skip-guidelines", action="store_true",
                        help="Skip the initial guidelines understanding step")
    args = parser.parse_args()
    
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
    
    process_all_scenes(args.video_folder, model, processor, skip_guidelines=args.skip_guidelines)
    
    del model
    del processor
    gc.collect()
    torch.cuda.empty_cache()
    print("Resources freed. Process complete.")

if __name__ == "__main__":
    main()