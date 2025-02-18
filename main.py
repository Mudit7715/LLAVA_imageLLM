import torch
import gradio as gr
from transformers import BitsAndBytesConfig, pipeline
import whisper
import time
import warnings
import os
from gtts import gTTS
from PIL import Image
import numpy as np
import datetime
import nltk
import locale
import re
import requests
import base64
from google.colab import userdata
from huggingface_hub import login

# Install necessary libraries (moved outside for clarity)
# !pip install -q -U transformers==4.37.2  # transformers module
# !pip install -q bitsandbytes==0.41.3 accelerate==0.25.0
# !pip install -q git+https://github.com/openai/whisper.git  # importing whisper module form github
# !pip install -q gradio  # for user interface and temporary deployment
# !pip install -q gTTS  # Google text-to-speech software


## Configuration and Setup

# Quantization Configuration
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,  # setting the quantization of the model to 4-bit
    bnb_4bit_compute_dtype=torch.float16,  # data type of the numbers used for computation. Helps in reducing the computational overhead
)

# Hugging Face Login
try:
    token = userdata.get('hf_token')
    hf_token = token
    login(token=hf_token)
except:
    print("HF_TOKEN not found in Colab secrets.  Please add it for authentication.")

# Model ID
model_id = "llava-hf/llava-1.5-7b-hf"

# Pipeline Initialization (Attempt with error handling)
try:
    pipe = pipeline("image-to-text",
                    model=model_id,
                    model_kwargs={"quantization_config": quantization_config})
except Exception as e:
    print(f"Error initializing pipeline: {e}")
    pipe = None  # or handle the error appropriately

# NLTK Punkt Sentence Tokenizer
try:
    nltk.download('punkt')
    from nltk import sent_tokenize
except Exception as e:
    print(f"Error downloading or importing nltk punkt: {e}")
    sent_tokenize = None  # or handle the error

# Locale Setting
try:
    locale.getlocale()
except Exception as e:
    print(f"Error getting locale: {e}")

# Device Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using torch {torch.__version__} ({DEVICE})")

# Whisper Model Loading
try:
    model = whisper.load_model("medium", device=DEVICE)
    print(
        f"Model is {'multilingual' if model.is_multilingual else 'English-only'} "
        f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
    )
except Exception as e:
    print(f"Error loading Whisper model: {e}")
    model = None

# Logger file
tstamp = datetime.datetime.now()  # Gets the current date and time
tstamp = str(tstamp).replace(' ', '_')  # replaces the spaces in the time stamp with '_' to make it easier to convert it into a filename
logfile = f'{tstamp}_log.txt'  # Creation of the filename with the timestamp and _log.txt

def writehistory(text):
    with open(logfile, 'a', encoding='utf-8') as f:
        f.write(text)
        f.write('\n')
    f.close()


## Functions

def img2txt(input_text, input_image):
    """
    Generates a text description of an image using the LLaVA model.
    """
    if pipe is None:
        return "LLaVA pipeline not initialized."

    try:
        # load the image
        image = Image.open(input_image)

        writehistory(f"Input text: {input_text} - Type: {type(input_text)} - Dir: {dir(input_text)}")  ## this is used to log the input_text content , datatype and available attributes

        if type(input_text) == tuple:
            prompt_instructions = """
            Describe the image using as much detail as possible, is it a painting, a photograph, what colors are predominant, what is the image about?
            """
        else:
            prompt_instructions = """
            Act as an expert in imagery descriptive analysis, using as much detail as possible from the image, respond to the following prompt:
            """ + input_text

        writehistory(f"prompt_instructions: {prompt_instructions}")  # logs the complete prompt instructions sent to llava model concatenated with the input
        prompt = "USER: \\n" + prompt_instructions + "\\nASSISTANT:"

        outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})  # output generation

        # Properly extract the response text.
        # In cases where the response or output by the model is not in the appropriate syntax or structure, this help in handling those exceptions.

        if outputs is not None and len(outputs[0]["generated_text"]) > 0:
            match = re.search(r'ASSISTANT:\\s*(.*)', outputs[0]["generated_text"])
            if match:
                # Extract the text after "ASSISTANT:"
                reply = match.group(1)
            else:
                reply = "No response found."
        else:
            reply = "No response generated."

        return reply

    except Exception as e:
        print(f"Error in img2txt: {e}")
        return f"Error processing image: {e}"

def transcribe(audio):
    """
    Transcribes audio to text using the Whisper model.
    """
    if model is None:
        return '', '', None  # Return empty strings and None audio file
    try:
        # Check if the audio input is None or empty
        if audio is None or audio == '':
            return ('', '', None)  # Return empty strings and None audio file

        # language = 'en'

        audio = whisper.load_audio(audio)
        audio = whisper.pad_or_trim(audio)  # Ensuring the audio data has proper length and padding for consistent inputs

        mel = whisper.log_mel_spectrogram(audio).to(model.device)

        _, probs = model.detect_language(mel)  # Inbuilt whisper functionality to recognise the language used by the speaker.

        options = whisper.DecodingOptions()
        result = whisper.decode(model, mel, options)  # transcripting audio
        result_text = result.text  # Taking the text output from the program and feeding it to a variable

        return result_text
    except Exception as e:
        print(f"Error in transcribe: {e}")
        return f"Transcription error: {e}"

def text_to_speech(text, file_path):
    """
    Converts text to speech using Google Text-to-Speech.
    """
    try:
        language = 'en'

        audioobj = gTTS(text=text,
                        lang=language,
                        slow=False)

        audioobj.save(file_path)

        return file_path
    except Exception as e:
        print(f"Error in text_to_speech: {e}")
        return f"TTS error: {e}"

# Set preferred locale encoding (if not already set)
try:
    locale.getpreferredencoding()
except:
    locale.getpreferredencoding = lambda: "UTF-8"  # setting the preffered locale encoding to UTF-8 same as the text and the audio input

## Creating a temporary audio file which can be used as a placeholder for audio processing in the model

# This command uses FFmpeg to generate a silent MP3 audio file named "Temp.mp3".
# !ffmpeg -f lavfi -i anullsrc=r=44100:cl=mono -t 10 -q:a 9 -acodec libmp3lame Temp.mp3

# Modified ffmpeg command execution
try:
    os.system("ffmpeg -f lavfi -i anullsrc=r=44100:cl=mono -t 10 -q:a 9 -acodec libmp3lame Temp.mp3")
except Exception as e:
    print(f"Error executing ffmpeg: {e}")


## Gradio Interface

def process_inputs(audio_path, image_path):
    """
    Processes audio and image inputs, combining speech-to-text and image description.
    """
    try:
        # Process the audio file (assuming this is handled by a function called 'transcribe')
        speech_to_text_output = transcribe(audio_path)

        # Handle the image input
        if image_path:
            chatgpt_output = img2txt(speech_to_text_output, image_path)
        else:
            chatgpt_output = "No image provided."

        # Assuming 'transcribe' also returns the path to a processed audio file
        processed_audio_path = text_to_speech(chatgpt_output, "Temp3.mp3")  # Replace with actual path if different

        return speech_to_text_output, chatgpt_output, processed_audio_path

    except Exception as e:
        print(f"Error in process_inputs: {e}")
        return f"Processing error: {e}", "", None  # Return error messages

# Create the interface
iface = gr.Interface(
    fn=process_inputs,
    inputs=[
        gr.Audio(sources=["microphone"], type="filepath"),
        gr.Image(type="filepath")
    ],
    outputs=[
        gr.Textbox(label="Speech to Text"),
        gr.Textbox(label="LLaVA Output"),
        gr.Audio("Temp.mp3")
    ],
    title="Deep Learning Mini Project 220968094",
    description="Upload an image and interact via voice input and audio response."
)

# Launch the interface
iface.launch(debug=True)
