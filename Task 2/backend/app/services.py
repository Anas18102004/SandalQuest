import os
import numpy as np
import torch
import speech_recognition as sr
from pydub import AudioSegment
from app.utils import tokenize_text, generate_answer, speak_answer
from app.config import OUTPUT_DIR

# Constants
SUPPORTED_AUDIO_FORMAT = "wav"

def convert_to_wav(file_path: str) -> str:
    """
    Convert the audio file to .wav format if necessary.
    Returns the path of the converted .wav file.
    """
    if file_path.endswith(f".{SUPPORTED_AUDIO_FORMAT}"):
        return file_path  # Already in .wav format

    try:
        audio = AudioSegment.from_file(file_path)
        wav_file_path = file_path.rsplit(".", 1)[0] + ".wav"
        audio.export(wav_file_path, format=SUPPORTED_AUDIO_FORMAT)
        print(f"Converted audio file to .wav format: {wav_file_path}")
        return wav_file_path
    except Exception as e:
        print(f"Error converting file to .wav format: {e}")
        return None

def recognize_audio(file_path: str) -> str:
    """
    Recognize speech from an audio file using Google Speech Recognition.
    Ensures the audio is in .wav format before processing.
    Returns recognized text if successful, None otherwise.
    """
    # Convert to .wav if needed
    file_path = convert_to_wav(file_path)
    if not file_path:
        return None

    # Initialize recognizer
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(file_path) as source:
            audio = recognizer.record(source)
        # Attempt recognition with Google API
        result = recognizer.recognize_google(audio)
        print(f"Recognition result: {result}")
        return result
    except sr.UnknownValueError:
        print("Speech recognition could not understand the audio.")
        return None
    except sr.RequestError as e:
        print(f"Request error in speech recognition: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during recognition: {e}")
        return None

def get_embeddings(tokens, model, device):
    """
    Generate embeddings for tokens using a transformer model.
    The function returns the mean embedding of the last hidden state for each token.
    """
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        # Move tokens to the specified device and generate embeddings
        tokens = tokens.to(device)
        outputs = model.transformer(tokens)
        mean_embedding = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
    return mean_embedding

def calculate_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two embeddings.
    Returns a float value between -1 and 1.
    """
    # Ensure embeddings are numpy arrays
    embedding1 = np.array(embedding1)
    embedding2 = np.array(embedding2)
    similarity = np.dot(embedding1, embedding2.T) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    return similarity
