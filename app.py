import gradio as gr
from generate import load_model, generate

model, vocab, gen_transform, device = load_model()


def predict(prompt_text):
    if not prompt_text.strip():
        return ""
    return generate(prompt_text, model, vocab, gen_transform, device)


with gr.Blocks(title="AI Text Generator") as demo:
    gr.Markdown("# Text Generator")

    text_input = gr.Textbox(label="Prompt", placeholder="Type something to start...", lines=3)
    output = gr.Textbox(label="Generated continuation", lines=5)

    generate_btn = gr.Button("Generate")
    generate_btn.click(fn=predict, inputs=text_input, outputs=output)

if __name__ == "__main__":
    demo.launch()