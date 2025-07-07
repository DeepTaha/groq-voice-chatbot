# app.py
import os
import whisper
from groq import Groq
from gtts import gTTS
import gradio as gr
import uuid

# Constants
MODEL_NAME = "llama3-70b-8192"

# Load Whisper model once
whisper_model = whisper.load_model("base")

def process_audio(audio_filepath):
    # Step 1: Transcribe with Whisper
    result = whisper_model.transcribe(audio_filepath)
    user_input = result["text"]

    # Step 2: Get response from Groq
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
    if not GROQ_API_KEY:
        raise ValueError("❌ GROQ_API_KEY not found. Please set it in Hugging Face Secrets.")

    client = Groq(api_key=GROQ_API_KEY)
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": user_input}],
        model=MODEL_NAME
    )
    bot_reply = chat_completion.choices[0].message.content

    # Step 3: Text-to-Speech
    tts = gTTS(text=bot_reply, lang='en')
    response_audio_path = f"{uuid.uuid4().hex}_response.mp3"
    tts.save(response_audio_path)

    return user_input, response_audio_path

# Gradio Interface
iface = gr.Interface(
    fn=process_audio,
    inputs=gr.Audio(type="filepath", label="Upload your voice (.wav/.mp3)"),
    outputs=[
        gr.Textbox(label="Transcribed Text"),
        gr.Audio(label="AI Response")
    ],
    title="🎤 Groq AI Voice Assistant",
    description="Upload your voice file. It will be transcribed using Whisper, replied to by Groq LLaMA 3, and spoken back using Google TTS.",
)

if __name__ == "__main__":
    iface.launch()
