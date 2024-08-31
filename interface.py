from firecrawl import FirecrawlApp
from openai import OpenAI
import pathlib as pl
import gradio as gr
import importlib
import os

# --------Set up Firecrawl, OpenAI and local folder--------
prompt_path = pl.Path("prompt")
prompt_list = list(prompt_path.glob("*.py"))
prompt_list = [str(prompt).split("/")[-1].split(".")[0] for prompt in prompt_list]
fire_app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))
css = """footer {visibility: hidden}
.logo img {height:100px; width:auto; margin:0 auto;}
"""
num_thread = 5
open_source_model = ""
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
p = pl.Path("data")
if not p.exists():
    p.mkdir()
r = pl.Path("results")
if not r.exists():
    r.mkdir()
# ----------------------------------------------------------


def process_url_markdown(url):
    scrape_result = fire_app.scrape_url(
        url, params={"formats": ["markdown"], "excludeTags": ["a", "img", "video"]}
    )
    return scrape_result["markdown"]


def process_url_json(url, model_type, prompt_type):
    scrape_result = fire_app.scrape_url(
        url, params={"formats": ["markdown"], "excludeTags": ["a", "img", "video"]}
    )["markdown"]
    # prompt_module = importlib.import_module(f"prompt.{prompt_type}")
    # llm_prompt = prompt_module.llm_prompt
    # query = llm_prompt.format(scrape_result)
    prompt_module = importlib.import_module(f"prompt.{prompt_type}")
    response_format = prompt_module.response_format
    if model_type == "Proprietary":
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at summarizing car review articles in JSON format.",
                },
                {
                    "role": "user",
                    "content": "Extract the following information in JSON format: "
                    + scrape_result,
                },
            ],
            max_tokens=16384,
            temperature=0.8,
            response_format=response_format,
        )
        print(response.choices[0].message.content)
        return response.choices[0].message.content
    elif model_type == "Open Source":
        return "Not supported yet."


def process_url(url, mode, model_type, prompt_type):
    if mode == "Markdown":
        return process_url_markdown(url), None
    elif mode == "JSON":
        return None, process_url_json(url, model_type, prompt_type)


def set_uninteractive(url):
    return gr.update(interactive=False)


def set_interactive(url):
    return gr.update(interactive=True)


def set_visible_and_interactive(mode):
    if mode == "Markdown":
        return (
            gr.update(visible=False, interactive=False),
            gr.update(visible=False, interactive=False),
            gr.update(visible=True),
            gr.update(visible=False),
        )
    elif mode == "JSON":
        return (
            gr.update(visible=True, interactive=True),
            gr.update(visible=True, interactive=True),
            gr.update(visible=False),
            gr.update(visible=True),
        )


with gr.Blocks(css=css, title="JSON Extractor") as app:
    with gr.Row():
        logo_img = gr.Image(
            "https://web.faa.illinois.edu/app/uploads/sites/14/2022/12/University-Wordmark-Full-Color-RGB-1-1200x0-c-default.webp",
            elem_classes="logo",
            show_download_button=False,
            show_label=False,
            container=False,
        )
    with gr.Row():
        gr.Markdown("# JSON Extractor")
    with gr.Row():
        with gr.Column(scale=1):
            mode = gr.Radio(
                ["Markdown", "JSON"],
                label="Output Format",
                info="Choose the output format",
                value="Markdown",
            )
            model_type = gr.Radio(
                ["Proprietary", "Open Source"],
                label="Model Type",
                info="Choose the model type",
                value="Proprietary",
                visible=False,
            )
            prompt_type = gr.Dropdown(
                choices=prompt_list,
                label="Object Type",
                info="Choose the object type that you want to extract information from",
                visible=False,
            )
            input_url = gr.Textbox(label="Input URL to be processed")
            submit_button = gr.Button("Submit")
        with gr.Column(scale=2):
            output_markdown = gr.Markdown(visible=True)
            output_json = gr.JSON(visible=False)

    mode.change(
        set_visible_and_interactive,
        inputs=mode,
        outputs=[model_type, prompt_type, output_markdown, output_json],
    )
    submit_button.click(
        set_uninteractive, inputs=input_url, outputs=submit_button
    ).then(
        process_url,
        inputs=[input_url, mode, model_type, prompt_type],
        outputs=[output_markdown, output_json],
    ).then(set_interactive, inputs=input_url, outputs=submit_button)

if __name__ == "__main__":
    app.queue(num_thread)
    app.launch(
        server_name="0.0.0.0",
        max_threads=num_thread,
        favicon_path="./assets/favicon.png",
        server_port=8080,
        ssl_verify=False,
    )
