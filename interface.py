import concurrent.futures
from tqdm import tqdm
import pathlib as pl
import gradio as gr
import requests
import json
import csv


css = """footer {visibility: hidden}
.logo img {height:100px; width:auto; margin:0 auto;}
"""
num_thread = 1
p = pl.Path('data')
if not p.exists():
    p.mkdir()
r = pl.Path('results')
if not r.exists():
    r.mkdir()

with gr.Blocks(css=css, title='SPIN Demo') as app:
    with gr.Row():
        logo_img=gr.Image('https://web.faa.illinois.edu/app/uploads/sites/14/2022/12/University-Wordmark-Full-Color-RGB-1-1200x0-c-default.webp',elem_classes='logo', show_download_button=False, show_label=False, container=False)
    with gr.Row():
        gr.Markdown('# SPIN Demo')
    

if __name__ == "__main__":
    app.queue(200)
    app.launch(
        server_name='0.0.0.0',
        max_threads=num_thread,
        favicon_path='./assets/favicon.png',
        server_port=8080,
        ssl_verify=False
        )