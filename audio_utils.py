from mutagen import File
from mutagen.id3 import ID3, TIT2
import numpy as np
import soundfile as sf
import librosa
import logging

def load_audio(file_path, **kwargs):
    try:
        return librosa.load(file_path, **kwargs)
    except Exception as e:
        logging.error(f"Error loading audio file {file_path}: {e}")
        return None, None

def split_audio(file_path, start_sample, end_sample, sr):
    """ Split the audio file at the specified sample range """
    y, _ = load_audio(file_path, sr=None, offset=start_sample/sr, duration=(end_sample-start_sample)/sr)
    return y   

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