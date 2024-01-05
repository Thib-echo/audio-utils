from datetime import datetime
import logging
import json
from json import JSONDecodeError
import re 
from pathlib import Path
import shutil

def read_transcription_data(transcription_file):
    """
    Read transcription data from a JSON file.

    Args:
    transcription_file (Path): Path to the audio file. The function constructs the path to the corresponding JSON file by replacing the audio file extension with '.json'.

    Returns:
    list: A list of transcription data segments. Returns an empty list if the file does not exist or in case of an error.
    """

    encodings = ['utf-8', 'ISO-8859-1', 'windows-1252']  # List of encodings to try

    try:
        if transcription_file.exists():
            for encoding in encodings:
                try:
                    with open(transcription_file, 'r', encoding=encoding) as f:
                        return json.load(f)
                except UnicodeDecodeError:
                    continue  # If this encoding fails, try the next one
            logging.error(f"Could not decode {transcription_file} with any known encoding.")
        else:
            logging.warning(f"Transcription file {transcription_file} does not exist.")
    except JSONDecodeError as e:
        logging.error(f"Error decoding JSON from {transcription_file}: {e}")
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
    
def format_filename(timestamp, file_path):
    file_name_components = [
        str(timestamp.year),
        str(timestamp.month),
        str(timestamp.day),
        str(timestamp.hour),
        str(timestamp.minute),
        str(timestamp.second),
        file_path.stem.split('_')[-1]
    ]
    file_name = '_'.join(file_name_components)

    return f"{file_name}{file_path.suffix}"

def parse_timestamp_from_filename(filename):
    # Parse the start timestamp from the file name
    file_name_pattern = r"(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)"
    match = re.match(file_name_pattern, filename.stem)
    if match:
        year, month, day, hour, minute, second = map(int, match.groups())
        timestamp = datetime(year, month, day, hour, minute, second)
    else:
        raise ValueError("File name does not match expected pattern")

    return timestamp

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

def clean_processed_files(directory):
    base_path = Path(directory)

    for subdir in base_path.iterdir():
        if subdir.is_dir():
            txt_file_path = subdir / f'{subdir.name}_transcription.txt'
            json_file_path = subdir / f'{subdir.name}_segments_data.json'

            # Check if txt file is empty
            if txt_file_path.stat().st_size == 0:
                shutil.rmtree(subdir)
                print(f"Deleted {subdir} as txt file is empty.")
                continue

            # Try reading the json file and handle UnicodeDecodeError
            try:
                with open(json_file_path, 'r', encoding='utf-8') as json_file:
                    json.load(json_file)
            except UnicodeDecodeError:
                shutil.rmtree(subdir)
                print(f"Deleted {subdir} due to UnicodeDecodeError in json file.")
            except JSONDecodeError:
                print(f"JSONDecodeError in {json_file_path}, but not deleting the directory.")
