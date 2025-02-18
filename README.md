# LLAVA_imageLLM
Using a local MultiModal LLM Llava for image-to-voice conversions

This phase focuses on the integration of speech-to-text, text generation, and text-to-speech components using the following software:
●	Bitsandbytes for Model Quantization
Overview: Bitsandbytes is used for model quantization, a technique that reduces the memory footprint of large language models by utilizing lower-precision number formats (e.g., 4-bit instead of 32-bit). This allows the model to fit into the limited memory of the Colab environment.
Applications: It is applied in this project to load the LLaVA model with 4-bit quantization.
Pros:
o	Reduces memory usage, enabling large models to run on smaller hardware configurations.
Cons:
o	Quantization might affect the model’s accuracy compared to full precision.

●	Accelerate for Hardware Optimization
Overview: Accelerate is a library designed to optimize large models across different hardware setups, including multiple GPUs. It helps manage the model’s parts and assigns them to the appropriate hardware for better performance.
Applications: Accelerate facilitates the loading of the quantized LLaVA model and ensures efficient use of available hardware resources.
Pros:
o	Supports multi-GPU setups and improves speed and efficiency in model deployment.
Cons:
o	Requires advanced configuration for optimal performance on specific hardware.

Whisper for Speech-to-Text
Overview: Whisper is a state-of-the-art automatic speech recognition (ASR) model developed by OpenAI. It is designed to transcribe multilingual speech and handle a wide range of accents, background noise, and technical speech, using a large and diverse dataset for training. Whisper is capable of transcribing speech with high accuracy across various languages and acoustic conditions.
Applications: In this project, Whisper can be applied to convert raw audio inputs into text transcriptions, especially when handling more complex or varied audio data. This forms the foundation for further text generation and analysis by models like Llama.
●	Provides high accuracy in speech recognition across multiple languages and noisy environments.
●	Handles diverse audio inputs, including accents, overlapping speech, and different audio qualities.
gTTS for Text-to-Speech
Overview: Google Text-to-Speech (gTTS) is a lightweight text-to-speech tool that is used to convert text into speech with minimal latency.
Applications: It is used in this project to convert generated text responses into speech for interactive audio-based outputs.
Pros:
o	Low latency and minimal resource requirements.
Cons:
o	Limited customization options for speech output in terms of voice and intonation.
Final Models for Implementation
●	Speech-to-text: Whisper model for transcribing audio to text.
●	Text Generation: LLaVA model for generating contextually relevant responses.
●	Text-to-Speech: gTTS for converting text-based outputs into speech.

Metrics
Hyperparameter tuning :
1.	Model quantization from 16-bits to 4-bits using Bitsandbytes library
2.	Setting a datatype for the quantised model to reduce computational overhead using “ bnb_4bit_compute_dtype=torch.float16”
3.	Setting device to CUDA GPU for easier computation  DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
4.	Using mel_spectogram for audio images for better human voice perception.

References
1.	https://www.gradio.app/docs/gradio/interface
2.	https://llava-vl.github.io/
3.	https://pypi.org/project/gTTS/
4.	https://github.com/openai/whisper
5.	https://pypi.org/project/bitsandbytes/
6.	https://huggingface.co/docs/accelerate/en/index

