from datetime import datetime
import numpy as np
from mutagen import File
from mutagen.id3 import ID3, TIT2
import soundfile as sf
from pathlib import Path
import librosa
import logging

import json
from json import JSONDecodeError

def read_transcription_data(file_path):
    """
    Read transcription data from a JSON file.

    Args:
    file_path (str): Path to the audio file. The function constructs the path to the corresponding JSON file by replacing the audio file extension with '.json'.

    Returns:
    list: A list of transcription data segments. Returns an empty list if the file does not exist or in case of an error.
    """

    try:
        file_path = Path(file_path)
        transcription_file = file_path.with_stem(file_path.stem + '_segments_data').with_suffix('.json')
        if transcription_file.exists():
            with open(transcription_file, 'r') as f:
                return json.load(f)
        else:
            logging.warning(f"Transcription file {transcription_file} does not exist.")
        return []
    except JSONDecodeError as e:
        logging.error(f"Error decoding JSON from {transcription_file}: {e}")
        return []
    except Exception as e:
        logging.error(f"Unexpected error while reading {transcription_file}: {e}")
        return []

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
    

def load_audio(file_path, **kwargs):
    try:
        return librosa.load(file_path, **kwargs)
    except Exception as e:
        logging.error(f"Error loading audio file {file_path}: {e}")
        return None, None
    
def save_audio_segment(segment_file_path, audio_segment, sr):
    try:
        sf.write(segment_file_path, audio_segment, sr)
    except Exception as e:
        logging.error(f"Error saving audio segment to {segment_file_path}: {e}")

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