"""
Transcribe + Analyze Pipeline
1. Whisper transcribes audio → timestamped segments
2. Formats transcript for LLM consumption
3. Sends to FuxiAPI for analysis

Usage: python transcribe.py [video_path]
"""
import whisper
import asyncio
import json
import sys
import os
from fuxi_api import FuxiAPI

### Helper functions
def transcribe_video(file_path, model_name="large-v3-turbo"):
    """Transcribe video/audio with Whisper and return formatted segments."""
    print(f"🎙️ Loading Whisper ({model_name})...")
    model = whisper.load_model(model_name)
    
    print(f"📝 Transcribing: {file_path}")
    result = model.transcribe(file_path, language="en")
    
    return result

def time_to_seconds(time_str):
    # Strip out any accidental brackets the LLM might include (e.g., "[00:15]")
    clean_time = time_str.replace('[', '').replace(']', '').strip()
    try:
        m, s = clean_time.split(':')
        return int(m) * 60 + int(s)
    except ValueError:
        print(f"⚠️ Warning: Could not parse timestamp '{time_str}'. Defaulting to 0.")
        return 0

def format_transcript_for_llm(result):
    """
    Convert Whisper output into a clean, timestamped transcript string 
    that an LLM can easily read and reference.
    """
    lines = []
    for seg in result["segments"]:
        start = seg["start"]
        end = seg["end"]
        text = seg["text"].strip()
        
        # Format as [MM:SS - MM:SS] text
        start_fmt = f"{int(start // 60):02d}:{int(start % 60):02d}"
        end_fmt = f"{int(end // 60):02d}:{int(end % 60):02d}"
        lines.append(f"[{start_fmt} - {end_fmt}] {text}")
    
    transcript = "\n".join(lines)
    return transcript


def save_transcript(transcript, output_path):
    """Save formatted transcript to a text file."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(transcript)
    print(f"💾 Transcript saved to: {output_path}")
### End of Helper functions

async def analyze_with_llm(transcript, prompt=None, analysis_mode="conservative"):
    """Send the transcript to FuxiAPI for analysis."""
    api = FuxiAPI()
    # 2. Define the specialized personas and tasks
    POLITICAL_PROMPTS = {
        "conservative": (
            "You are an expert political strategist specializing in conservative messaging.\n"
            "Task: Highlight the following segments of this speech to align strongly with Republican and conservative values, "
            "supporting Donald Trump's policy agenda (e.g., economic nationalism, border security). \n"
            "Then, highlight the three most impactful segments from your rewrite and provide strategic reasons "
            "why they will resonate with a conservative voter base."
        ),
        "liberal": (
            "You are an expert political strategist specializing in progressive messaging.\n"
            "Task: Highlight the following segments of this speech to align strongly with Democratic and liberal values, "
            "critiquing or opposing Donald Trump's policy agenda (e.g., climate change, wealth equality). \n"
            "Then, highlight the three most impactful segments from your rewrite and provide strategic reasons "
            "why they will resonate with a progressive voter base."
        )
    }

    json_format = (
        "\n\nSTYLE GUIDELINE FOR TITLE:\n"
        "The 'title' field MUST be a highly engaging, TikTok-style 'hook' caption. "
        "Be creative and highly strategic based on your assigned persona. "
        "Use punctuation (like quotation marks or bolding) to reframe statements (e.g., to emphasize a point or imply skepticism/sarcasm). "
        "You may include emojis to actively show strong support, outrage, or disbelief, perfectly matching your political alignment.\n\n"
        "You MUST STRICTLY format your entire response strictly as a valid JSON object. "
        "Do not include any markdown, preamble, or conversational text. "
        "Use this exact schema:\n"
        "{\n"
        '  "summary": "A concise summary of the overall speech",\n'
        '  "key_topics": ["Topic 1", "Topic 2"],\n'
        '  "highlights": [\n'
        "    {\n"
        '      "title": "Your TikTok-style hook with emojis and framing punctuation",\n'
        '      "start_timestamp": "00:00", // Exact MM:SS string from the transcript\n'
        '      "end_timestamp": "00:33",   // Exact MM:SS string from the transcript\n'
        '      "rationale": "Your strategic reason for choosing this segment"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
    )

    # 3. Build the final instruction block dynamically (overwriting any previous memory)
    if analysis_mode in POLITICAL_PROMPTS:
        # Apply the selected political bias and enforce an output format
        base_instruction = POLITICAL_PROMPTS[analysis_mode]
        format_requirements = (
            "\n\nPlease format your response exactly as follows:\n"
            "1. **Top 3 Highlighted Segments & Rationale**\n"
            "2. **Key Topics**\n"
            "3. **Summary**\n\n"
        )
        INSTRUCTION = base_instruction + json_format
    else:
        # Fallback to the Neutral/Bipartisan default
        INSTRUCTION = (
            "You are a NEUTRAL, BIPARTISAN political speech analyst. "
            "Please analyze the speech and provide:\n"
            "1. **Key Topics** — Main themes discussed, with relevant timestamps\n"
            "2. **Notable Quotes** — Important or viral-worthy statements\n"
            "3. **Highlights** — The most engaging/important moments with timestamps\n"
            "4. **Summary** — A concise summary of the entire speech\n\n"
        )
        
    # 4. Inject the transcript
    FINAL_PAYLOAD = f"{INSTRUCTION}--- TRANSCRIPT ---\n{transcript}\n--- END TRANSCRIPT ---"

    # --- Test 1: Execute the API Call ---
    #print(f"\n📝 Test 1: get_response ({analysis_mode} mode)")

    # Assuming 'output' was a preamble string from your original code, prepend it here if needed
    response = await api.get_response(FINAL_PAYLOAD) 
    #print(f"Response:\n{response}")

    api.close()
    return response


async def main(): 
    # Default path
    file_path = "/mnt/c/Users/dexter.neo/Downloads/trump.mp4"
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    
    # Step 1: Transcribe
    result = transcribe_video(file_path)
    
    # Step 2: Format for LLM
    transcript = format_transcript_for_llm(result)
    print(f"\n📄 Transcript ({len(result['segments'])} segments):\n")
    print(transcript[:500] + "..." if len(transcript) > 500 else transcript)
    
    # Step 3: Save transcript
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    save_transcript(transcript, f"{base_name}_transcript.txt")
    
    # Step 4: Save raw Whisper output
    with open(f"{base_name}_whisper_raw.json", "w", encoding="utf-8") as f:
        json.dump(result["segments"], f, indent=2, ensure_ascii=False)
    print(f"💾 Raw segments saved to: {base_name}_whisper_raw.json")
    
    # Step 5: Analyze with LLM
    analysis = await analyze_with_llm(transcript)
    print(f"\n{'='*60}")
    print("🧠 LLM ANALYSIS:")
    print(f"{'='*60}")
    print(analysis)
    
    # Save analysis
    with open(f"{base_name}_analysis.txt", "w", encoding="utf-8") as f:
        f.write(analysis)
    print(f"\n💾 Analysis saved to: {base_name}_analysis.txt")


if __name__ == "__main__":
    asyncio.run(main())