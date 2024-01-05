
from pydub import AudioSegment, silence
from mutagen.mp3 import MP3
from pathlib import Path
import pandas as pd
from utils import *
from audio_utils import *
import datetime
import argparse
import librosa
import logging
import shutil
import json
import re
from transcribe_all import process_audio_files

# Constants
END_OF_CONV_WORDS = ["au revoir", "bon courage", "bonne soirée", "bonne jounrée", "bye", "goodbye"]
START_OF_CONV_WORDS = ["bonjour", "bon jour", "hello", "hi", "good morning"]
DATA_FOLDER = Path("../audio_database/ch30_test/raw_transcription/")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Placeholder for argument parsing
parser = argparse.ArgumentParser(description='Audio Processing and Transcription Analysis Script')
parser.add_argument("--source_folder", type=str, help="Path to the folder containing audio files and transcription")
# parser.add_argument("--dest_folder", type=str, help="Path to the folder where processed audio files will be saved along their transcription")


### IDENTIFY AND SPLIT MERGED FILES ###
def identify_and_split_merged_files(transcription_folder, processed_folder, gap_threshold=15, start_word_cooldown=15, split_type='mp3'):
    """
    Process audio files by identifying merged files, splitting them, and copying to new location.

    Args:
    folder_path (str): Path to the folder containing audio files.
    splited_files_folder (Path): Path to the folder for storing processed files.
    gap_threshold (int): Time threshold in seconds to identify gaps indicating separate conversations.
    start_word_cooldown (int): Cooldown time to identify start of a new conversation.
    """ 

    for audio_segments_data in transcription_folder.rglob('*.json'):
        transcription_data = read_transcription_data(audio_segments_data)
        is_merged, merged_details = check_if_merged(transcription_data, gap_threshold, start_word_cooldown)
        
        file = audio_segments_data.with_name(audio_segments_data.stem.rsplit('_', 2)[0] + ".mp3")

        if is_merged:
            logging.info(f"File {file.stem} detected as merged, will be splitted in {len(merged_details) + 1} segments")
            split_merged_file(file, transcription_data, merged_details, processed_folder, split_type)
        else:
            shutil.copytree(audio_segments_data.parent, processed_folder / audio_segments_data.parent.name, dirs_exist_ok=True)
            
def check_if_merged(transcription_data, gap_threshold, start_word_cooldown):
    last_end_conversation_time = None
    last_start_conversation_time = None
    merged_details = []

    for segment in transcription_data:
        segment_id = segment['id']
        segment_text = segment['text'].lower().strip()
        segment_start = float(segment['start'])

        # Check for end-of-conversation words
        for end_word in END_OF_CONV_WORDS:
            if re.search(r'\b' + re.escape(end_word) + r'\b', segment_text):
                last_end_conversation_time = segment_start
                break

        for start_word in START_OF_CONV_WORDS:
            if re.search(r'\b' + re.escape(start_word) + r'\b', segment_text):
                if segment_start >= 2:  # assuming 5 seconds as a threshold for the beginning
                    time_since_last_end = segment_start - last_end_conversation_time if last_end_conversation_time is not None else float('inf')
                    time_since_last_start = segment_start - last_start_conversation_time if last_start_conversation_time is not None else float('inf')
                    if time_since_last_end <= gap_threshold and time_since_last_start > start_word_cooldown:
                        merged_details.append((segment_start, start_word, max(segment_id - 1, 0)))
                        last_start_conversation_time = segment_start
                        break

    return bool(merged_details), merged_details

def adjust_json_data(json_data, time_offset, start_id):
    for i, segment in enumerate(json_data):
        segment['id'] = start_id + i
        segment['start'] -= time_offset
        segment['end'] -= time_offset
        for word in segment['words']:
            word['start'] -= time_offset
            word['end'] -= time_offset
    return json_data

def split_merged_file(file_path, transcription_data, cutting_times, dest_folder_path, split_type):
    if split_type == 'mp3':
        y, sr = load_audio(file_path, sr=None)

        # Initial start sample for the first segment
        prev_cut_sample = 0
        original_stem = file_path.stem
        for _, (cut_time, _) in enumerate(cutting_times):
            cut_sample = int(cut_time * sr)
            audio_segment = split_audio(file_path, prev_cut_sample, cut_sample, sr)

            # Determine new file name and title
            audio = MP3(file_path)
            title = audio['TIT2'][0] if 'TIT2' in audio else 'Unknown'
            start_timestamp, end_timestamp = parse_timestamp_from_title(title)
            new_start_timestamp = start_timestamp + pd.to_timedelta(prev_cut_sample / sr, unit='s')
            new_end_timestamp = start_timestamp + pd.to_timedelta(cut_sample / sr, unit='s')
            new_file_name = format_filename(new_start_timestamp, original_stem)
            new_title = f"{new_start_timestamp.strftime('%d/%m/%Y %H:%M:%S')} - {new_end_timestamp.strftime('%d/%m/%Y %H:%M:%S')}"

            # Save the audio segment
            segment_file_path = dest_folder_path / new_file_name
            if (segment_file_path.parent / segment_file_path.stem).exists():
                logging.info(f"Segment {segment_file_path} already transcribed")
            elif segment_file_path.exists():
                logging.info(f"Segment {segment_file_path} already saved")
            else:
                save_audio_segment(segment_file_path, audio_segment, sr)
                # Update metadata for the segment
                copy_metadata(file_path, segment_file_path, new_title)

            prev_cut_sample = cut_sample

        # Handle the last segment from the last cut to the end of the file
        last_segment = split_audio(file_path, prev_cut_sample, len(y), sr)
        new_start_timestamp = start_timestamp + pd.to_timedelta(prev_cut_sample / sr, unit='s')
        new_file_name = format_filename(new_start_timestamp, original_stem)
        last_segment_file_path = dest_folder_path / new_file_name
        new_title = f"{new_start_timestamp.strftime('%d/%m/%Y %H:%M:%S')} - {end_timestamp.strftime('%d/%m/%Y %H:%M:%S')}"

        if (last_segment_file_path.parent / last_segment_file_path.stem).exists():
            logging.info(f"Segment {segment_file_path} already saved")
        else:
            save_audio_segment(last_segment_file_path, last_segment, sr)
            # Update metadata for the last segment
            copy_metadata(file_path, last_segment_file_path, new_title)

     # Handle text files
    elif split_type == 'txt':
        start_timestamp = parse_timestamp_from_filename(file_path)

        current_segment_id = 0
        prev_cut_time = 0.0

        for i, (cut_time, _, cut_segment_id) in enumerate(cutting_times):
            current_transcript = [seg['text'] for seg in transcription_data[current_segment_id:cut_segment_id]]
            current_json = transcription_data[current_segment_id:cut_segment_id]

            new_start_timestamp = start_timestamp + datetime.timedelta(seconds=prev_cut_time)
            new_file_name_base = format_filename(new_start_timestamp, file_path)
            new_text_file_name = new_file_name_base if i != 0 else file_path.name
            new_text_file_name = Path(new_text_file_name).with_name(Path(new_text_file_name).stem + '_transcription.txt')
            new_json_file_name = Path(new_file_name_base).with_name(Path(new_file_name_base).stem + '_segments_data.json')

            split_folder_path = Path(dest_folder_path) / Path(new_file_name_base).stem
            split_folder_path.mkdir(exist_ok=True)

            save_transcript_segment(split_folder_path, new_text_file_name, current_transcript)

            adjusted_json = adjust_json_data(current_json, prev_cut_time, 1)
            save_json_segment(split_folder_path, new_json_file_name, adjusted_json)

            prev_cut_time = cut_time
            current_segment_id = cut_segment_id

        remaining_transcript = [seg['text'] for seg in transcription_data[current_segment_id:]]
        remaining_json = transcription_data[current_segment_id:]

        new_start_timestamp = start_timestamp + datetime.timedelta(seconds=prev_cut_time)
        last_file_name_base = format_filename(new_start_timestamp, file_path)
        last_text_file_name = Path(last_file_name_base).with_name(Path(last_file_name_base).stem + '_transcription.txt')
        last_json_file_name = Path(last_file_name_base).with_name(Path(last_file_name_base).stem + '_segments_data.json')

        split_folder_path = Path(dest_folder_path) / Path(last_file_name_base).stem
        split_folder_path.mkdir(exist_ok=True)

        save_transcript_segment(split_folder_path, last_text_file_name, remaining_transcript)
        adjusted_json = adjust_json_data(remaining_json, prev_cut_time, 1)
        save_json_segment(split_folder_path, last_json_file_name, adjusted_json)

def save_transcript_segment(dest_folder_path, file_name, transcript_segment):
    """
    Save a segment of the transcript to a file.
    """
    segment_file_path = Path(dest_folder_path) / file_name
    with open(segment_file_path, 'w', encoding="utf-8") as segment_file:
        segment_file.write(' '.join(transcript_segment))

def save_json_segment(dest_folder_path, file_name, json_segment):
    """
    Save a segment of the JSON data to a file.
    """
    segment_file_path = Path(dest_folder_path) / file_name
    with open(segment_file_path, 'w', encoding="utf-8") as segment_file:
        json.dump(json_segment, segment_file, indent=4)

### CREATE AUDIO DATABASE ###
def check_for_problematic_end(transcription_data, words, end_time, density_threshold=0.2, diversity_threshold=0.5):
    # Find time of the last occurrence of the end_word
    last_end_word_time = None
    for segment in reversed(transcription_data):
        if any(re.search(r'\b' + re.escape(end_word) + r'\b', segment['text'].lower()) for end_word in words):
            last_end_word_time = segment['end']
            break

    if last_end_word_time is None:
        return False

    # Analyze word density and diversity after the last end word
    remaining_segments = [seg for seg in transcription_data if seg['start'] >= last_end_word_time]
    total_words = sum(len(seg['text'].split()) for seg in remaining_segments)
    unique_words = len(set(word for seg in remaining_segments for word in seg['text'].split()))

    remaining_time = end_time - last_end_word_time
    word_density = total_words / remaining_time if remaining_time > 0 else 0
    word_diversity = unique_words / total_words if total_words > 0 else 1

    return word_density > density_threshold or word_diversity < diversity_threshold

def create_audio_database(dest_folder, split_type):
    """
    Create a database of audio files with metadata and transcription data.

    Args:
    folder_path (str): Path to the folder containing audio files and their transcription data.

    Returns:
    pandas.DataFrame: A DataFrame containing metadata and transcription data for each audio file.
    """
    files = [audio_file for audio_file in Path(dest_folder).rglob(f'*.{split_type}') ]
    data = []

    for file in files:
        try:
            if split_type == 'mp3':
                audio = MP3(file)
                title = audio['TIT2'][0] if 'TIT2' in audio else 'Unknown'
                start_timestamp, end_timestamp = parse_timestamp_from_title(title)
                audio_length = audio.info.length

                transcription_path = file.with_stem(file.stem + '_transcription').with_suffix('.txt')
                transcription_data_path = file.with_name(file.stem + '_segments_data.json')
                transcription_data = read_transcription_data(transcription_data_path)

                is_start_file = check_word_in_timeframe(transcription_data, START_OF_CONV_WORDS, 0, 10)
                
                is_end_file = check_word_in_timeframe(transcription_data, END_OF_CONV_WORDS, max(0, audio_length - 10), audio_length, is_end_segment=True)
                is_problematic_end = check_for_problematic_end(transcription_data, END_OF_CONV_WORDS, audio_length, density_threshold=0.25, diversity_threshold=0.5)
                is_end_file = is_end_file or is_problematic_end

                is_complete = is_start_file and is_end_file

                # If the file is complete, set is_start_file and is_end_file to False
                if is_complete:
                    is_start_file = False
                    is_end_file = False

                data.append({
                    "File Name": file.stem,
                    "File Path": str(file),
                    "Transcription Path": str(transcription_path) if transcription_path.exists() else None,
                    "Start Timestamp": start_timestamp,
                    "End Timestamp": end_timestamp,
                    "Audio Length": audio_length,
                    "Is End File": is_end_file,
                    "Is Start File": is_start_file,
                    "Is Complete": is_complete,
                    "Precedent File": None,
                    "Next File": None
                })
            elif split_type == 'txt':
                transcription_data_path = file.with_name('_'.join(file.stem.split('_')[:-1]) + '_segments_data.json')
                transcription_data = read_transcription_data(transcription_data_path)
                
                audio_length = transcription_data[-1]['end']

                start_timestamp = parse_timestamp_from_filename(file)
                end_timestamp = start_timestamp + datetime.timedelta(seconds=audio_length)

                is_start_file = check_word_in_timeframe(transcription_data, START_OF_CONV_WORDS, 0, 10)
                
                is_end_file = check_word_in_timeframe(transcription_data, END_OF_CONV_WORDS, max(0, audio_length - 10), audio_length, is_end_segment=True)
                is_problematic_end = check_for_problematic_end(transcription_data, END_OF_CONV_WORDS, audio_length, density_threshold=0.25, diversity_threshold=0.5)
                is_end_file = is_end_file or is_problematic_end

                is_complete = is_start_file and is_end_file

                # If the file is complete, set is_start_file and is_end_file to False
                if is_complete:
                    is_start_file = False
                    is_end_file = False

                data.append({
                    "File Name": file.stem,
                    "File Path": str(file),
                    "Transcription Path": str(file) if file.exists() else None,
                    "Start Timestamp": start_timestamp,
                    "End Timestamp": end_timestamp,
                    "Audio Length": audio_length,
                    "Is End File": is_end_file,
                    "Is Start File": is_start_file,
                    "Is Complete": is_complete,
                    "Precedent File": None,
                    "Next File": None
                })
        except Exception as e:
            print(f"Error processing file {file}: {e}")

    return pd.DataFrame(data)

### FIND ASSOCIATED FILES AND MERGE SEPARATE CONVERSATIONS ###
def find_associated_files(df, time_delta=20):
    """
    Identify groups of audio files that form complete conversations based on their labels, timestamps, and time delta.

    Args:
    df (pandas.DataFrame): DataFrame with audio file metadata, including labels and timestamps.
    time_delta (int): Maximum time difference in seconds to consider files associated.

    Returns:
    list of tuples: Each tuple contains paths of associated audio files forming a conversation.
    """
    # Sort DataFrame by timestamps
    df_sorted = df.sort_values(by=['Start Timestamp']).reset_index(drop=True)

    associated_files = []
    current_group = []
    grouped_files = set()

    def time_diff_okay(current_end, next_start):
        return 0 <= (next_start - current_end).total_seconds() <= time_delta
    
    def get_channel(file_path):
        # Extract the channel number from the file path
        match = re.search(r'ch(\d+)', file_path)
        return match.group(1) if match else None
    
    for index, row in df_sorted.iterrows():
        file_path = row['File Path']
        if file_path in grouped_files:
            continue

        file_channel = get_channel(file_path)

        label = None
        if row['Is Start File']:
            label = 'start'
        elif row['Is End File']:
            label = 'end'
        elif row['Is Complete']:
            continue  # Complete files do not need to be grouped
        else:
            label = 'other'

        if label == 'start':
            # If a new start is found, save the previous group and start a new one
            if current_group:
                associated_files.append(tuple(current_group))
            current_group = [file_path]
        elif label == 'end':
            # End file - add to the current group and close the group
            current_group.append(file_path)
            associated_files.append(tuple(current_group))
            current_group = []
        else:  # label == 'other'
            # Other files - add to the current group if within time delta and same channel
            if current_group:
                last_index = df_sorted.index[df_sorted['File Path'] == current_group[-1]].tolist()[0]
                last_row = df_sorted.iloc[last_index]
                last_file_channel = get_channel(current_group[-1])

                if last_file_channel == file_channel and time_diff_okay(last_row['End Timestamp'], row['Start Timestamp']):
                    current_group.append(file_path)
                else:
                    associated_files.append(tuple(current_group))
                    current_group = [file_path]
            else:
                current_group.append(file_path)

    # Add the last group if not empty
    if current_group:
        associated_files.append(tuple(current_group))

    return associated_files

def concatenate_texts(text_files):
    concatenated_text = ''
    for file_path in text_files:
        with open(file_path, 'r', encoding='utf-8') as file:
            concatenated_text += file.read() + ' '
    return concatenated_text

def merge_jsons(json_files):
    merged_json = []
    current_id = 1
    for file_path in json_files:
        with open(file_path, 'r', encoding="utf-8") as file:
            data = json.load(file)
            for segment in data:
                segment['id'] = current_id
                current_id += 1
                merged_json.append(segment)
    return merged_json

def merge_associated_files(associated_files, df, dest_folder, split_type):
    """
    Merge associated audio files and save it on the raw_audio_folder

    Args:
    associated_files (list of tuples): Each tuple contains file paths of associated audio files.
    df (pandas.DataFrame): A DataFrame containing audio file metadata.
    raw_audio_folder (str): Path to the folder where the grouped and merged audio files will be stored.

    This function processes each group of associated files, merges their audio content, and saves the merged audio in the specified folder. It also updates the metadata for the merged files based on the group's start and end timestamps.
    """

    for group in associated_files:
        # Create a unique folder for each group
        if split_type == 'mp3':

            audio_files = []
            start_timestamp = None
            end_timestamp = None

            for file_path in group:        
                # Add file to list for merging
                audio_files.append(file_path)

                # Extract timestamps for merged file metadata
                file_info = df[df['File Path'] == file_path].iloc[0]
                if start_timestamp is None or file_info['Start Timestamp'] < start_timestamp:
                    start_timestamp = file_info['Start Timestamp']
                if end_timestamp is None or file_info['End Timestamp'] > end_timestamp:
                    end_timestamp = file_info['End Timestamp']

            # Merge and save audio
            fused_audio, sample_rate = merge_audios(audio_files)
            fused_file_path = Path(dest_folder) / Path(audio_files[0]).name
            save_audio_segment(fused_file_path, fused_audio, sample_rate)
            
            # Copy metadata from the beginning file to the merged file
            start_timestamp = start_timestamp.strftime("%d/%m/%Y %H:%M:%S")
            end_timestamp = end_timestamp.strftime("%d/%m/%Y %H:%M:%S")

            new_title = f"{start_timestamp} - {end_timestamp}"
            copy_metadata(group[0], fused_file_path, new_title)
        
        elif split_type == 'txt':
            # Logic for merging text and JSON files
            text_files = [Path(file_path) for file_path in group]
            json_files = [file_path.with_name('_'.join(file_path.stem.split('_')[:-1]) + '_segments_data.json') for file_path in text_files]

            merged_text = concatenate_texts(text_files)
            merged_json = merge_jsons(json_files)

            merged_folder = Path(dest_folder) / '_'.join(text_files[0].name.split('_')[:-1]) 
            merged_folder.mkdir(exist_ok=True, parents=True)

            merged_text_path = merged_folder / text_files[0].name
            merged_json_path = merged_text_path.with_name('_'.join(merged_text_path.stem.split('_')[:-1]) + '_segments_data.json')

            with open(merged_text_path, 'w', encoding='utf-8') as file:
                file.write(merged_text)
            save_json_segment(merged_folder, merged_json_path.name, merged_json)

    # # Delete old transcription folder
    for group in associated_files:
        for file_path in group:
            shutil.rmtree(Path(file_path).parent)

def main(args):
    source_folder = Path(args.source_folder)
    transcription_folder = source_folder.parent / "raw_transcriptions"
    processed_folder = source_folder.parent / "processed_files"

    split_type = 'txt'

    # 1. Initial transcription of raw_audio into raw_transcriptions
    # process_audio_files(source_folder, transcription_folder, move=False)

    # 2. Identifying and splitting merged files
    identify_and_split_merged_files(transcription_folder, processed_folder, split_type=split_type)

    clean_processed_files(processed_folder)
    
    # 3. Transcriptions of newly created audio files
    # process_audio_files(processed_folder, processed_folder, move=True)
    

    # 4. Creation of the dataframe and labelling of the audios (beginning of file, end, complete file or other)    
    df_audio = create_audio_database(processed_folder, split_type=split_type)

    # 5. Find associated audio files
    associated_files = find_associated_files(df_audio)

    # 6. Merge of associated files
    merge_associated_files(associated_files, df_audio, processed_folder, split_type=split_type)

    # # 7. Final transcriptions of newly created audio files
    # process_audio_files(processed_folder, processed_folder, move=True)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)