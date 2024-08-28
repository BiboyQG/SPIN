import gradio as gr
import time

counter = 1

def visible_component(input_text):
    return gr.update(visible=True)


def generate_output(input_text):
    #gr.update(output_text,visible=True)
    global counter
    time.sleep(2)
    output_text = "Hello, " + input_text + "!"
    counter += 1
    return output_text

with gr.Blocks() as demo:
    with gr.Row():
    
        # column for inputs
        with gr.Column():
            input_text = gr.Textbox(label="Input Text")
            submit_button = gr.Button("Submit")
                   
        # column for outputs
        with gr.Column():
            output_text = gr.Textbox(visible=False)
            
    submit_button.click(
        fn=visible_component,
        inputs=input_text,
        outputs=output_text
    ).then(
        #time.sleep(2),
        fn=generate_output,
        inputs=input_text,
        outputs=output_text
        )

demo.launch()