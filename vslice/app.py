import gradio as gr
import time
import os
import spaces

@spaces.GPU
def mock_summarize(video_file, prompt, progress=gr.Progress()):
    if not video_file:
        raise gr.Error("Please upload a video to test the demo. / 请上传视频进行演示。")
        
    progress(0.1, desc="Extracting frames... / 提取视频帧...")
    time.sleep(0.5)
    progress(0.4, desc="Graph-DPO Model Scoring... / Graph-DPO 模型打分...")
    time.sleep(1.0)
    progress(0.7, desc="Extracting Highlights... / 提取精彩片段...")
    time.sleep(0.5)
    progress(0.9, desc="Assembling Final Edit... / 组合最终剪辑...")
    time.sleep(0.5)
    
    return video_file

fig_misalignment = "./assets/fig1_misalignment_real.png"
fig_reliability = "./assets/fig1_reliability_split.png"
fig_bin_strength = "./assets/fig2_bin_strength.png"
fig_posthoc = "./assets/fig4_posthoc_graph_failure.png"

# Placeholders for example Videos
video_example1 = "./assets/69118fe052fb155119d76733j1I4g7PP06.mp4"
video_example2 = "./assets/safe_1771730634.mp4"
video_example3 = "./assets/safe_1771730634.mp4"

def build_content(lang):
    if lang == "EN":
        gr.Markdown("# ✂️🎬 VSLICE: Binary Preference Alignment for Open-Ended Video Summarization with Large Vision-Language Models")
        gr.Markdown("Welcome to the interactive project showcase for VSLICE! Scroll down to understand the motivation, the algorithm, and see example outputs.")
        
        # ---------------------------------------------------------
        # SECTION 0: Example Outputs (Gallery)
        # ---------------------------------------------------------
        gr.Markdown("---")
        gr.Markdown("## Demo Gallery (Summary Outputs)")
        gr.Markdown("Examples of video summaries that VSLICE can create.")
        
        with gr.Row():
            if os.path.exists(video_example1):
                gr.Video(value=video_example1, label="Sports Example", interactive=False, autoplay=True, elem_classes=["gallery-video"])
            else:
                gr.Markdown(f"*(Video placeholder 1: {video_example1})*")

            if os.path.exists(video_example2):
                gr.Video(value=video_example2, label="Politics Example", interactive=False, autoplay=True, elem_classes=["gallery-video"])
            else:
                gr.Markdown(f"*(Video placeholder 2: {video_example2})*")
                
            if os.path.exists(video_example3):
                gr.Video(value=video_example3, label="Video Games Example", interactive=False, autoplay=True, elem_classes=["gallery-video"])
            else:
                gr.Markdown(f"*(Video placeholder 3: {video_example3})*")

        # ---------------------------------------------------------
        # SECTION 1: Motivation
        # ---------------------------------------------------------
        gr.Markdown("---")
        gr.Markdown("## 1. Introduction & Motivation")
        gr.Markdown("### The Problem: LVLM Misalignment & Overconfidence")
        gr.Markdown(
            "Although Large Vision-Language Models (LVLMs) possess strong semantic recognition ability, "
            "their raw confidence scores are poorly calibrated for highlight localization. Reliability diagrams reveal that they suffer from massive overconfidence "
            "on non-highlight background frames, manifesting as a persistent **Human Misalignment Gap**."
        )
        with gr.Row():
            # Left Image
            if os.path.exists(fig_misalignment):
                gr.Image(fig_misalignment, label="Highlight Misalignment", show_label=False, height=350)
            else:
                gr.Markdown(f"*(Image not found at {fig_misalignment})*")
            
            # Center Image
            if os.path.exists(fig_reliability):
                gr.Image(fig_reliability, label="Reliability Diagram (LVLM Overconfidence)", show_label=False, height=350)
            else:
                gr.Markdown(f"*(Image not found at {fig_reliability})*")

            # Right Image
            if os.path.exists(fig_bin_strength):
                gr.Image(fig_bin_strength, label="Bin-Strength Plot", show_label=False, height=350)
            else:
                gr.Markdown(f"*(Image not found at {fig_bin_strength})*")
        
        gr.Markdown("### The Post-Hoc Smoothing Trap")
        gr.Markdown(
            "A naive solution to temporal fragmentation is to apply a post-hoc graph or temporal smoothing to the raw logits. "
            "However, because the zero-shot policy is fundamentally misaligned, post-hoc smoothing simply **smears the overconfident false positives** "
            "across adjacent frames, completely destroying precision."
        )
        if os.path.exists(fig_posthoc):
            gr.Image(fig_posthoc, label="The Post-Hoc Graph Smoothing Failure", show_label=False)
        else:
            gr.Markdown(f"*(Image not found at {fig_posthoc})*")

        # ---------------------------------------------------------
        # SECTION 2: Algorithm
        # ---------------------------------------------------------
        gr.Markdown("---")
        gr.Markdown("## 2. Algorithm: Graph-DPO")
        gr.Markdown("### Direct Preference Optimization on Graphs")
        gr.Markdown("""
        To solve the alignment issue and temporal fragmentation natively, we introduce **Graph-DPO**.
        By injecting a visual-temporal similarity graph directly into the DPO training loop, the policy learns to output aligned, temporally coherent scores without smearing false positives.
        """)
        
        # ---------------------------------------------------------
        # SECTION 3: Demo
        # ---------------------------------------------------------
        gr.Markdown("---")
        gr.Markdown("## 3. Example Outputs (Demo)")
        gr.Markdown("### Interactive Summarization Demo")
        gr.Markdown(
            "Experience the final output of the Graph-DPO aligned model. "
            "Uploading a video will simulate the pipeline and output the example clip.*"
        )
        
        with gr.Row():
            with gr.Column():
                video_in = gr.Video(label="Input Long Video (e.g., video_11)")
                prompt_in = gr.Textbox(
                    label="User Prompt", 
                    value="Extract the most exciting action highlights from this video.",
                    lines=2
                )
                submit_btn = gr.Button("✨ Generate Summary", variant="primary")
                
            with gr.Column():
                video_out = gr.Video(label="Summarized Highlight Reel (Output)")
                
        submit_btn.click(
            fn=mock_summarize,
            inputs=[video_in, prompt_in],
            outputs=[video_out]
        )
        
    elif lang == "CN":
        gr.Markdown("# ✂️🎬 VSLICE: Binary Preference Alignment for Open-Ended Video Summarization with Large Vision-Language Models")
        gr.Markdown("欢迎来到 VSLICE 的互动项目展示！请向下滚动以了解项目动机、算法，并查看示例输出。")
        
        # ---------------------------------------------------------
        # SECTION 0: Example Outputs (Gallery)
        # ---------------------------------------------------------
        gr.Markdown("---")
        gr.Markdown("## 示例图库")
        gr.Markdown("将鼠标悬停在视频上即可收听音频。")
        
        with gr.Row():
            if os.path.exists(video_example1):
                gr.Video(value=video_example1, label="体育示例", interactive=False, autoplay=True, elem_classes=["gallery-video"])
            else:
                gr.Markdown(f"*(视频占位符 1: {video_example1})*")

            if os.path.exists(video_example2):
                gr.Video(value=video_example2, label="政治示例", interactive=False, autoplay=True, elem_classes=["gallery-video"])
            else:
                gr.Markdown(f"*(视频占位符 2: {video_example2})*")
                
            if os.path.exists(video_example3):
                gr.Video(value=video_example3, label="游戏示例", interactive=False, autoplay=True, elem_classes=["gallery-video"])
            else:
                gr.Markdown(f"*(视频占位符 3: {video_example3})*")

        # ---------------------------------------------------------
        # SECTION 1: Motivation
        # ---------------------------------------------------------
        gr.Markdown("---")
        gr.Markdown("## 1. 简介与动机")
        gr.Markdown("### 问题：LVLM 的不对齐与过度自信")
        gr.Markdown(
            "尽管大型视觉语言模型 (LVLM) 具有强大的语义识别能力，"
            "但它们的原始置信度分数在精彩片段定位方面的校准效果很差。它们在背景帧上表现出极度的过度自信，"
            "导致了巨大的**人类偏好不对齐差距 (Misalignment Gap)**。"
        )
        with gr.Row():
            # Left Image
            if os.path.exists(fig_misalignment):
                gr.Image(fig_misalignment, label="高亮不对齐分析", show_label=False, height=350)
            else:
                gr.Markdown(f"*(未找到图片：{fig_misalignment})*")
                
            # Right Image
            if os.path.exists(fig_reliability):
                gr.Image(fig_reliability, label="可靠性图 (LVLM 过度自信)", show_label=False, height=350)
            else:
                gr.Markdown(f"*(未找到图片：{fig_reliability})*")
        
        gr.Markdown("### 事后平滑陷阱")
        gr.Markdown(
            "解决时间碎片化的一种天真方法是对原始逻辑值进行事后的图平滑或时间平滑。"
            "然而，由于零样本策略在根本上就是不对齐的，事后平滑只会**将过度自信的误报涂抹**"
            "到相邻帧上，从而彻底破坏准确率。"
        )
        if os.path.exists(fig_posthoc):
            gr.Image(fig_posthoc, label="事后图平滑失效", show_label=True)
        else:
            gr.Markdown(f"*(未找到图片：{fig_posthoc})*")

        # ---------------------------------------------------------
        # SECTION 2: Algorithm
        # ---------------------------------------------------------
        gr.Markdown("---")
        gr.Markdown("## 2. 算法：Graph-DPO")
        gr.Markdown("### 图上的直接偏好优化 (DPO)")
        gr.Markdown("""
        为了从根本上解决对齐问题和时间碎片化，我们引入了 **Graph-DPO**。
        通过将视觉-时间相似度图直接注入到 DPO 训练循环中，策略学习输出对齐的、时间连贯的分数，而不会涂抹误报。
        """)
        
        # ---------------------------------------------------------
        # SECTION 3: Demo
        # ---------------------------------------------------------
        gr.Markdown("---")
        gr.Markdown("## 3. 示例输出 (演示)")
        gr.Markdown("### 交互式摘要演示")
        gr.Markdown(
            "体验 Graph-DPO 对齐模型的最终输出。"
            "上传视频将模拟处理管道并输出示例剪辑。*"
        )
        
        with gr.Row():
            with gr.Column():
                video_in = gr.Video(label="输入长视频 (例如，video_11)")
                prompt_in = gr.Textbox(
                    label="用户提示词", 
                    value="提取这个视频中最激动人心的动作精彩片段。",
                    lines=2
                )
                submit_btn = gr.Button("✨ 生成摘要", variant="primary")
                
            with gr.Column():
                video_out = gr.Video(label="摘要精彩片段集锦 (输出)")
                
        submit_btn.click(
            fn=mock_summarize,
            inputs=[video_in, prompt_in],
            outputs=[video_out]
        )

theme = gr.themes.Glass(
    primary_hue="sky", 
    neutral_hue="slate",
    radius_size="lg",
    text_size=gr.themes.sizes.text_lg,
    font=[gr.themes.GoogleFont("Inconsolata"), "Arial", "sans-serif"]
).set(
    # Backgrounds
    body_background_fill="#0f172a",
    body_background_fill_dark="#0f172a",
    block_background_fill="#1e293b",
    block_background_fill_dark="#1e293b",
    background_fill_primary="#0f172a",
    background_fill_primary_dark="#0f172a",

    # Body text
    body_text_color="#f1f5f9",
    body_text_color_dark="#f1f5f9",

    # Block / component labels (e.g. "Input Long Video", "Output")
    block_label_background_fill="#334155",
    block_label_background_fill_dark="#334155",
    block_label_text_color="#f1f5f9",
    block_label_text_color_dark="#f1f5f9",
    block_title_text_color="#f1f5f9",
    block_title_text_color_dark="#f1f5f9",

    # Input / Textbox
    input_background_fill="#0f172a",
    input_background_fill_dark="#0f172a",
    input_border_color="#334155",
    input_border_color_dark="#334155",
    input_placeholder_color="#64748b",

    # Radio buttons (language toggle)
    checkbox_label_background_fill="#1e293b",
    checkbox_label_background_fill_dark="#1e293b",
    checkbox_label_text_color="#f1f5f9",
    checkbox_label_text_color_dark="#f1f5f9",
    checkbox_label_border_color="#334155",
    checkbox_label_border_color_dark="#334155",
    checkbox_label_background_fill_selected="#0284c7",
    checkbox_label_background_fill_selected_dark="#0284c7",
    checkbox_label_text_color_selected="#ffffff",
    checkbox_label_text_color_selected_dark="#ffffff",
    checkbox_label_border_color_selected="#38bdf8",
    checkbox_label_border_color_selected_dark="#38bdf8",

    # Primary button (✨ Generate Summary)
    button_primary_background_fill="#0284c7",
    button_primary_background_fill_dark="#0284c7",
    button_primary_background_fill_hover="#0369a1",
    button_primary_background_fill_hover_dark="#0369a1",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",
    button_primary_border_color="#38bdf8",
    button_primary_border_color_dark="#38bdf8",

    # Secondary button (default)
    button_secondary_background_fill="#1e293b",
    button_secondary_background_fill_dark="#1e293b",
    button_secondary_background_fill_hover="#334155",
    button_secondary_background_fill_hover_dark="#334155",
    button_secondary_text_color="#f1f5f9",
    button_secondary_text_color_dark="#f1f5f9",
    button_secondary_border_color="#334155",
    button_secondary_border_color_dark="#334155",
)

with gr.Blocks(theme=theme) as demo:
    demo.load(None, None, None)
    
    # Top right language toggle
    with gr.Row():
        with gr.Column(scale=9):
            gr.Markdown("") # Spacer to push the toggle to the right
        with gr.Column(scale=1, min_width=150):
            lang_toggle = gr.Radio(
                choices=["ENG", "中文"], 
                value="ENG", 
                show_label=False, 
                container=False,
                interactive=True
            )
            
    with gr.Column(visible=True) as en_group:
        build_content("EN")
        
    with gr.Column(visible=False) as cn_group:
        build_content("CN")
        
    def switch_lang(lang):
        if "EN" in lang:
            return gr.update(visible=True), gr.update(visible=False)
        else:
            return gr.update(visible=False), gr.update(visible=True)
            
    lang_toggle.change(
        fn=switch_lang, 
        inputs=lang_toggle, 
        outputs=[en_group, cn_group]
    )

if __name__ == "__main__":
    demo.launch()