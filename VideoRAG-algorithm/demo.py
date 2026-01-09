import os
import logging
import argparse
import warnings
import multiprocessing
from videorag._llm import openai_4o_mini_config
from videorag import VideoRAG, QueryParam

# Setup logging
warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.WARNING)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed of generator")
    parser.add_argument("--language", type=str, default="ENG", help="Analysis Language")
    parser.add_argument("--prompt", type=str, default="Describe this video in detail.")
    args = parser.parse_args()
    return args

def main(args):
    # 1. Initialize QwenVideoRAG
    # The working_dir is where the vector index and metadata will be saved.
    working_dir = "./videorag-workdir"
    print(f"Initializing Qwen VideoRAG in {working_dir}...")

    videorag = VideoRAG(
        llm=openai_4o_mini_config,
        working_dir=working_dir
    )

    # 2. Indexing Phase
    # TODO: Update these paths to actual video files on your machine.
    video_paths = [
        #'./vids/trump.mp4'
        #'./vids/69118fe052fb155119d76733j1I4g7PP06.mp4',
        #'./vids/6911901d0b52d480daffcdf5OKs2mEUO06.mp4'
        './vids/2025-12-23 13-44-19_Trim.mp4'
    ]

    # Check if files exist before processing to avoid vague errors
    valid_paths = [p for p in video_paths if os.path.exists(p)]
    if not valid_paths:
        print("Error: No valid video files found. Please check 'video_paths' list.")
        return

    print(f"Indexing {len(valid_paths)} videos... (This may take a while depending on GPU)")
    videorag.insert_video(video_path_list=valid_paths)
    print("Indexing complete.")

    # 3. Querying Phase
    # We load the specific caption/VLM model required for retrieval/generation
    print("Loading caption model...")
    videorag.load_caption_model(debug=False)

    query_text = args.prompt
    
    print(f"Querying: {query_text}")
    
    # Configure query parameters
    param = QueryParam(mode="videorag")
    # Set to False if you want timestamps/clip references in the output
    param.wo_reference = True 

    response = videorag.query(query=query_text, param=param)
    
    print("-" * 30)
    print("Response:")
    print(response)
    print("-" * 30)

if __name__ == '__main__':
    args = get_args()
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        # Ignore if it was already set
        pass
        
    main(args)