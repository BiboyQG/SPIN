from firecrawl import FirecrawlApp
import pathlib as pl
import gradio as gr
import os

fire_app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))
num_thread = 1
css = """footer {visibility: hidden}
.logo img {height:100px; width:auto; margin:0 auto;}
"""
p = pl.Path('data')
if not p.exists():
    p.mkdir()
r = pl.Path('results')
if not r.exists():
    r.mkdir()

def process_url(url):
    scrape_result = fire_app.scrape_url(url, params={"formats": ["markdown"], "excludeTags": ["a", "img", "video"]})
    return scrape_result['markdown']

def set_uninteractive(url):
    return gr.update(interactive=False)

def set_interactive(url):
    return gr.update(interactive=True)

with gr.Blocks(css=css, title='SPIN Demo') as app:
    with gr.Row():
        logo_img=gr.Image('https://web.faa.illinois.edu/app/uploads/sites/14/2022/12/University-Wordmark-Full-Color-RGB-1-1200x0-c-default.webp',elem_classes='logo', show_download_button=False, show_label=False, container=False)
    with gr.Row():
        gr.Markdown('# SPIN Demo')
    with gr.Row():
        gr.Markdown('### Extract from HTML to Markdown')
    with gr.Row():
        with gr.Column(scale=1):
            input_url = gr.Textbox(label="Input URL to be processed")
            submit_button = gr.Button("Submit")
        with gr.Column(scale=3):
            output_markdown = gr.Markdown()

    submit_button.click(set_uninteractive, inputs=input_url, outputs=submit_button).then(process_url, inputs=input_url, outputs=output_markdown).then(set_interactive, inputs=input_url, outputs=submit_button)

if __name__ == "__main__":
    app.queue(num_thread)
    app.launch(
        server_name='0.0.0.0',
        max_threads=num_thread,
        favicon_path='./assets/favicon.png',
        server_port=8080,
        ssl_verify=False
        )