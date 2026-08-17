import gradio as gr
from generate import load_model, suggest_next_words

NUM_SUGGESTIONS = 5

model, vocab, gen_transform, device = load_model()


def get_suggestions(prompt_text):
    if not prompt_text.strip():
        return [gr.update(visible=False) for _ in range(NUM_SUGGESTIONS)]

    words = suggest_next_words(prompt_text, model, vocab, gen_transform, device, num_suggestions=NUM_SUGGESTIONS)

    updates = []
    for i in range(NUM_SUGGESTIONS):
        if i < len(words):
            full_text = f"{prompt_text.rstrip()} {words[i]}"
            updates.append(gr.update(value=full_text, visible=True))
        else:
            updates.append(gr.update(visible=False))
    return updates


def use_suggestion(full_text):
    return full_text


with gr.Blocks(title="AI Text Generator") as demo:
    gr.Markdown("# Text Generator")

    text_input = gr.Textbox(label="Prompt", placeholder="Start typing...", lines=3, autofocus=True)

    gr.Markdown("**Suggestions**")
    with gr.Row():
        suggestion_buttons = [gr.Button(visible=False) for _ in range(NUM_SUGGESTIONS)]

    text_input.change(fn=get_suggestions, inputs=text_input, outputs=suggestion_buttons)
    text_input.submit(fn=get_suggestions, inputs=text_input, outputs=suggestion_buttons)

    for btn in suggestion_buttons:
        btn.click(fn=use_suggestion, inputs=btn, outputs=text_input).then(
            fn=get_suggestions, inputs=text_input, outputs=suggestion_buttons
        )

if __name__ == "__main__":
    demo.launch()