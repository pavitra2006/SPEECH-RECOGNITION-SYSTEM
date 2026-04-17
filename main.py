import io
import os
from pathlib import Path

import librosa
import numpy as np
import streamlit as st
import torch

try:
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
except ImportError:
    from transformers.models.wav2vec2 import Wav2Vec2ForCTC, Wav2Vec2Processor

try:
    from pydub import AudioSegment
    from pydub.utils import which
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

# Load Wav2Vec2 model and processor
@st.cache_resource
def load_model():
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    return processor, model

processor, model = load_model()

SUPPORTED_AUDIO_EXTENSIONS = {
    "wav",
    "aiff",
    "aif",
    "flac",
    "mp3",
    "ogg",
    "m4a",
    "aac",
    "wma",
    "opus",
    "spx",
    "au",
    "aifc",
    "mpeg",
}

st.title("Speech-to-Text Transcription")
st.write("Upload an audio file to transcribe it to text (supports English).")

uploaded_file = st.file_uploader("Choose an audio file", type=None)


def convert_audio_to_wav(file_bytes: bytes, file_ext: str | None) -> io.BytesIO:
    if file_ext and file_ext.lower() in {"wav", "aiff", "aif", "flac"}:
        return io.BytesIO(file_bytes)

    if not PYDUB_AVAILABLE:
        raise RuntimeError(
            "Non-WAV audio conversion requires pydub. Install it with `pip install pydub`."
        )

    if which("ffmpeg") is None and which("ffmpeg.exe") is None:
        raise RuntimeError(
            "ffmpeg is required for audio format conversion. Install ffmpeg and ensure it is on your PATH."
        )

    audio_format = file_ext if file_ext else None
    audio_segment = AudioSegment.from_file(io.BytesIO(file_bytes), format=audio_format)
    wav_io = io.BytesIO()
    audio_segment.export(wav_io, format="wav")
    wav_io.seek(0)
    return wav_io


if uploaded_file is not None:
    st.audio(uploaded_file, format=uploaded_file.type or "audio/wav")

    if st.button("Transcribe"):
        with st.spinner("Transcribing..."):
            try:
                file_name = uploaded_file.name or "audio"
                file_ext = Path(file_name).suffix.lower().lstrip(".") or None
                if file_ext and file_ext not in SUPPORTED_AUDIO_EXTENSIONS:
                    st.warning(
                        f"File extension .{file_ext} is not commonly supported. "
                        "The app will still try to process it if possible."
                    )

                audio_data = uploaded_file.getvalue()
                audio_file = convert_audio_to_wav(audio_data, file_ext)
                
                # Load audio with librosa
                audio_file.seek(0)
                speech, sample_rate = librosa.load(audio_file, sr=16000)
                
                # Process with Wav2Vec2
                inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
                with torch.no_grad():
                    logits = model(inputs.input_values).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                text = processor.batch_decode(predicted_ids)[0]
                
                st.success("Transcription completed!")
                st.write("**Transcribed Text:**")
                st.write(text)

            except RuntimeError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"An error occurred: {e}")
else:
    st.info("Please upload an audio file to get started.")