import os
import torch
import logging
from tqdm import tqdm
from faster_whisper import WhisperModel
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

def speech_to_text(video_name, working_dir, segment_index2name, audio_output_format):    
    model_id = "openai/whisper-large-v3-turbo"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize the pipeline
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device=device,
        model_kwargs={"attn_implementation": "sdpa"}, # optimizations
    )

    transcripts = {}
    cache_path = os.path.join(working_dir, '_cache', video_name)

    for index in tqdm(segment_index2name, desc=f"Speech Recognition {video_name}"):
        segment_name = segment_index2name[index]
        audio_file = os.path.join(cache_path, f"{segment_name}.{audio_output_format}")

        if os.path.exists(audio_file):
            # Run inference
            # chunk_length_s handles long audio files automatically
            result = asr_pipeline(
                audio_file, 
                chunk_length_s=30, 
                batch_size=8,
                return_timestamps=True
            )
            
            text = result["text"].strip()

            transcripts[index] = text
            print(f"Segment {index}: {text[:50]}...")
        else:
            transcripts[segment_index] = ""

    print("The ASR transcripts:", transcripts)
    return transcripts

def old_speech_to_text(video_name, working_dir, segment_index2name, audio_output_format):
    model = WhisperModel(
        ".checkpoints/faster-distil-whisper-large-v3", 
        device='cpu',
        compute_type="int8",
        )
    model.logger.setLevel(logging.WARNING)
    
    cache_path = os.path.join(working_dir, '_cache', video_name)
    
    transcripts = {}
    for index in tqdm(segment_index2name, desc=f"Speech Recognition {video_name}"):
        segment_name = segment_index2name[index]
        audio_file = os.path.join(cache_path, f"{segment_name}.{audio_output_format}")

        # if the audio file does not exist, skip it
        if not os.path.exists(audio_file):
            transcripts[index] = ""
            continue
        
        segments, info = model.transcribe(audio_file)
        result = ""
        for segment in segments:
            result += "[%.2fs -> %.2fs] %s\n" % (segment.start, segment.end, segment.text)
        transcripts[index] = result
    
    return transcripts