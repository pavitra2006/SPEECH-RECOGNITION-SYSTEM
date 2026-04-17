# Speech-to-Text Project (TASK-2)
COMPANY : CODTECH IT SOLUTIONS NAME : PAVITRA SAVARAPU INTERN ID : CTIS7186 DOMAIN : Artificial Intelligence DURATION : 6 WEEKS MENTOR : NEELA SANTOSH
*DESCRIPTION*:
The main.py is a Streamlit-based speech-to-text transcription application that converts uploaded audio files into text using a pretrained transformer model.

### What the task does
- Offers a web interface for uploading audio files in multiple formats (WAV, MP3, FLAC, etc.)
- Automatically converts non-WAV files to WAV format for processing
- Transcribes the audio content into English text
- Displays the transcribed text in the browser
- Supports audio playback for verification

### Tools and architecture
- Streamlit: creates the interactive web app, handles file uploads, audio playback, and result display
- Hugging Face Transformers: loads `facebook/wav2vec2-base-960h` model and processor for speech recognition
- Audio processing pipeline: pydub for format conversion, librosa for loading and resampling audio
- Model inference: tokenizes audio features, runs through Wav2Vec2, decodes to text

### Libraries used
- `streamlit`
- `transformers`
- `librosa`
- `torch`
- `pydub` (optional, for audio conversion)
- `pathlib` and `io` for file handling

### How it works
1. User uploads an audio file via the Streamlit uploader
2. The app checks the file extension and converts to WAV if necessary (requires ffmpeg for non-WAV formats)
3. Audio is loaded with librosa at 16kHz sample rate
4. The Wav2Vec2 processor tokenizes the audio into input features
5. The model generates logits, which are decoded to text using argmax
6. The final transcription is displayed, with the original audio playable for reference

### Why it is useful
This project demonstrates an end-to-end speech recognition pipeline, combining audio preprocessing, deep learning inference, and a user-friendly web interface. It showcases practical NLP applications for accessibility, transcription services, and voice-enabled interfaces, making it ideal for prototyping speech-to-text solutions. The use of pretrained models ensures high accuracy without requiring extensive training data.


*OUTPUT*:
https://github.com/user-attachments/assets/4dafbf61-4c40-4aef-86d1-a9f83f7f8587

## CODTECH Internship Project

**Build a Basic Speech-to-Text System Using Pre-trained Models and Libraries like SpeechRecognition or Wav2Vec.**

### Description

This project implements a basic speech-to-text system capable of transcribing short audio clips. It uses the `SpeechRecognition` library with Google's Web Speech API to convert audio from WAV files into text.

### Features

- Transcribes audio from WAV files
- Uses Google's speech recognition service
- Handles errors gracefully
- Simple command-line interface

### Requirements

- Python 3.x
- SpeechRecognition library (install via `pip install -r requirements.txt`)

### Usage

1. Place your audio file as `sample.wav` in the project directory.
2. Run the script: `python main.py`
3. The transcription will be printed to the console.

### Files

- `main.py`: Main script for speech-to-text conversion
- `sample.wav`: Sample audio file (replace with your own)
- `requirements.txt`: Python dependencies
- `README.md`: This file

### Completion Certificate

A completion certificate will be issued on your internship end date.

CODTECH
