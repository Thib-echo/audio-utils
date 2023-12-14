from datetime import datetime
import logging
import json
from json import JSONDecodeError
import re 

def read_transcription_data(transcription_file):
    """
    Read transcription data from a JSON file.

    Args:
    transcription_file (Path): Path to the audio file. The function constructs the path to the corresponding JSON file by replacing the audio file extension with '.json'.

    Returns:
    list: A list of transcription data segments. Returns an empty list if the file does not exist or in case of an error.
    """

    try:
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
    
def format_filename(timestamp, original_stem):
    file_name_components = [
        str(timestamp.year),
        str(timestamp.month),
        str(timestamp.day),
        str(timestamp.hour),
        str(timestamp.minute),
        str(timestamp.second),
        original_stem.split('_')[-1]
    ]
    file_name = '_'.join(file_name_components)

    return f"{file_name}.mp3"

def check_words_in_transcription(transcription_path, words):
    try:
        with open(transcription_path, 'r') as file:
            transcription = file.read().lower()
            return [word for word in words if word in transcription]
    except FileNotFoundError:
        return []

def check_word_in_timeframe(transcription_data, words, start_time, end_time, is_end_segment=False):
    for segment in transcription_data:
        segment_text = segment['text'].lower()

        if is_end_segment:
            segment_time = segment['end']
        else:
            segment_time = segment['start']

        if start_time <= segment_time <= end_time:
            if any(re.search(r'\b' + re.escape(word) + r'\b', segment_text) for word in words):
                return True
    return False
