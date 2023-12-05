from datetime import datetime
import numpy as np
from mutagen import File
from mutagen.id3 import ID3, TIT2
import librosa

def parse_timestamp_from_title(title):
    try:
        timestamps = title.split('-')
        start_timestamp = datetime.strptime(timestamps[0].strip(), "%d/%m/%Y %H:%M:%S")
        end_timestamp = datetime.strptime(timestamps[1].strip(), "%d/%m/%Y %H:%M:%S")
        return start_timestamp, end_timestamp
    except (IndexError, ValueError):
        return None, None

def check_words_in_transcription(transcription_path, words):
    try:
        with open(transcription_path, 'r') as file:
            transcription = file.read().lower()
            return [word for word in words if word in transcription]
    except FileNotFoundError:
        return []
    

def load_audio(file_path):
    """ Load an audio file """
    audio, sr = librosa.load(file_path, sr=None)
    return audio, sr

def copy_metadata(src_file, dst_file, new_title=None):
    """ Copy metadata from src_file to dst_file """
    src_tags = File(src_file)
    dst_tags = ID3()

    for key, value in src_tags.items():
        dst_tags.add(value)

    if new_title:
        dst_tags.add(TIT2(encoding=3, text=new_title))

    dst_tags.save(dst_file)

def merge_audios(audio_files):
    """ Merge audio files """
    merged_audio = []
    sr = None

    for file in audio_files:
        audio, file_sr = load_audio(file)
        if sr is None:
            sr = file_sr
        elif sr != file_sr:
            raise ValueError("Sample rates of the audio files must match")
        merged_audio.append(audio)

    # Concatenate all audio files
    merged_audio = np.concatenate(merged_audio)

    return merged_audio, sr