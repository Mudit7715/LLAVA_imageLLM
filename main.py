import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
from transformers import BitsAndBytesConfig, pipeline
import whisper
import warnings
from gtts import gTTS
from PIL import Image
import numpy as np
import datetime
import re
from huggingface_hub import login
import nltk
import gradio as gr

# Configure warnings and locale
warnings.filterwarnings("ignore", category=UserWarning)
import locale
locale.getpreferredencoding = lambda: "UTF-8"

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model configurations
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

# Hugging Face authentication
hf_token = "enter your huggingface token here"
login(token=hf_token)

# Initialize models
model_id = "llava-hf/llava-1.5-7b-hf"
pipe = pipeline(
    "image-text-to-text", 
    model=model_id, 
    model_kwargs={
        "quantization_config": quantization_config,
        "device_map": "auto"
    },
    use_fast=True
)
whisper_model = whisper.load_model("medium", device=DEVICE)

# Download NLTK data
nltk.download('punkt' ,quiet=True)
nltk.download('punkt_tab' , quiet = True)
from nltk import sent_tokenize

# Logger setup
def get_logfile():
    # Replace colons with underscores for Windows compatibility
    tstamp = str(datetime.datetime.now()).replace(' ','_').replace(':','-')
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return os.path.join(log_dir, f'{tstamp}_log.txt')

def writehistory(text, logfile=None):
    try:
        if logfile is None:
            logfile = get_logfile()
        with open(logfile, 'a', encoding='utf-8') as f:
            f.write(f"{text}\n")
    except Exception as e:
        print(f"Warning: Could not write to log file: {e}")

def img2txt(input_text, input_image):
    image = Image.open(input_image)
    writehistory(f"Input text: {input_text} - Type: {type(input_text)} - Dir: {dir(input_text)}")
    
    prompt_instructions = """
    Describe the image using as much detail as possible, is it a painting, a photograph, what colors are predominant, what is the image about?
    """ if isinstance(input_text, tuple) else f"""
    Act as an expert in imagery descriptive analysis, using as much detail as possible from the image, respond to the following prompt:
    {input_text}"""
    
    writehistory(f"prompt_instructions: {prompt_instructions}")
    prompt = f"USER: <image>\n{prompt_instructions}\nASSISTANT:"
    
    outputs = pipe(image, text=prompt, generate_kwargs={"max_new_tokens": 200})
    
    if outputs and outputs[0]["generated_text"]:
        match = re.search(r'ASSISTANT:\s*(.*)', outputs[0]["generated_text"])
        return match.group(1) if match else "No response found."
    return "No response generated."

def transcribe(audio):
    if not audio:
        return ('', '', None)
    
    # Verify file exists
    if not os.path.exists(audio):
        print(f"Audio file not found: {audio}")
        return "Error: Audio file not found"
    
    try:
        audio_data = whisper.load_audio(audio)
        audio_data = whisper.pad_or_trim(audio_data)
        mel = whisper.log_mel_spectrogram(audio_data).to(whisper_model.device)
        
        options = whisper.DecodingOptions()
        result = whisper.decode(whisper_model, mel, options)
        return result.text
    except Exception as e:
        print(f"Error in transcription: {str(e)}")
        return f"Error: {str(e)}"

def text_to_speech(text, file_path):
    audioobj = gTTS(text=text, lang='en', slow=False)
    audioobj.save(file_path)
    return file_path

def process_inputs(audio_path, image_path):
    # Process the audio file
    if audio_path and os.path.exists(audio_path):
        speech_to_text_output = transcribe(audio_path)
    else:
        print(f"Warning: Audio file not found at {audio_path}")
        print("Available audio files in current directory:")
        for file in os.listdir():
            if file.endswith(('.wav', '.mp3', '.m4a')):
                print(f"- {file}")
        return "No audio input", "No audio transcription", None

    # Handle the image input
    if image_path and os.path.exists(image_path):
        chatgpt_output = img2txt(speech_to_text_output, image_path)
    else:
        chatgpt_output = "No image provided."

    processed_audio_path = text_to_speech(chatgpt_output, "Temp3.mp3")

    return speech_to_text_output, chatgpt_output, processed_audio_path

iface = gr.Interface(
    fn=process_inputs,
    inputs=[
        gr.Audio(sources=["microphone"], type="filepath"),
        gr.Image(type="filepath")
    ],
    outputs=[
        gr.Textbox(label="Speech to Text"),
        gr.Textbox(label="llava Output"),
        gr.Audio("Temp.mp3")
    ],
    title="Deep Learning Mini Project 220968094",
    description="Upload an image and interact via voice input and audio response."
)

# Launch the interface
iface.launch(debug=True,share=True)
