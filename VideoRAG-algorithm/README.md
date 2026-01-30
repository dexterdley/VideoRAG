# Naraka Highlight Detector (Video Slicing & Summarization)

This tool is a dedicated highlight detector for *Naraka: Bladepoint* gameplay videos. It leverages **MiniCPM-V 2.6** (a multimodal large language model) to visually analyze video frames and detect specific in-game events like "First Blood", "Triple Kill", or "Invincible".

When a highlight is detected, the tool automatically:
1.  **Slices** the relevant video segment (with context).
2.  **Stitches** overlapping clips into longer sequences.
3.  **Compiles** a final highlighting video (`full_highlights.mp4`).

## ✨ Features

-   **Visual Recognition**: Uses `MiniCPM-V-2_6-int4` to "see" and understand game content.
-   **Smart Slicing**: Uses `ffmpeg` stream copying for instant cuts without re-encoding quality loss.
-   **Auto-Stitching**: Merges adjacent highlights intelligently.
-   **Gradio UI**: Simple web interface to upload videos and view results.
-   **Custom Prompts**: Supports English and Chinese prompts for flexible detection.

## 🛠️ Installation

### 1. Environment Setup
Please use the provided `environment.yml` in the root directory to create the Conda environment.

```bash
# Navigate to the project root
cd ..
conda env create -f environment.yml
conda activate VSLICE
```

### 2. Additional Dependencies
The GUI requires `gradio` and `ffmpeg`, which might not be in the base environment.

```bash
pip install gradio
# Ensure ffmpeg is installed on your system (e.g., sudo apt install ffmpeg)
```

### 3. Model Download
The tool expects the **MiniCPM-V-2_6-int4** model to be present locally.
By default, it looks for the model in example:
-   `./MiniCPM-V-2_6-int4` (Windows/WSL relative path)

You can download it via Git LFS:
```bash
git lfs install
git clone https://huggingface.co/openbmb/MiniCPM-V-2_6-int4
```

## 🚀 Usage

1.  Activate the environment:
    ```bash
    conda activate VSLICE
    ```

2.  Run the GUI:
    ```bash
    python naraka_gui.py
    ```

3.  Open the browser link (usually `http://127.0.0.1:7860/?__theme=dark`).

4.  **Upload a Video**: Drag and drop your mp4 gameplay.
5.  **Click "Find Highlights"**: The model will scan the video.
    -   Progress is shown in the log.
    -   Detected clips will appear in the "Newest Hit" player.
    -   A compiled summary video will be generated at the end.


## 📂 Output
All detected clips and the final compilation are saved in the `highlights_found/` folder.
