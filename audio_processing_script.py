
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
parser.add_argument("--dest_folder", type=str, help="Path to the folder where processed audio files will be saved along their transcription")


### IDENTIFY AND SPLIT MERGED FILES ###
def identify_and_split_merged_files(source_folder_path, dest_folder_path, gap_threshold=15, start_word_cooldown=15):
    """
    Process audio files by identifying merged files, splitting them, and copying to new location.

    Args:
    folder_path (str): Path to the folder containing audio files.
    splited_files_folder (Path): Path to the folder for storing processed files.
    gap_threshold (int): Time threshold in seconds to identify gaps indicating separate conversations.
    start_word_cooldown (int): Cooldown time to identify start of a new conversation.
    """

    source_folder_path = Path(source_folder_path)
    dest_folder_path = Path(dest_folder_path)

    for audio_segments_data in source_folder_path.rglob('*.json'):
        transcription_data = read_transcription_data(audio_segments_data)
        is_merged, merged_details = check_if_merged(transcription_data, gap_threshold, start_word_cooldown)

        audio_file = audio_segments_data.with_name(audio_segments_data.stem.rsplit('_', 2)[0] + ".mp3")
        if is_merged:
            logging.info(f"File {audio_file.stem} detected as merged, will be splitted in {len(merged_details) + 1} segments")
            split_merged_file(audio_file, merged_details, dest_folder_path)
        else:
            shutil.copytree(audio_segments_data.parent, dest_folder_path / audio_segments_data.parent.name, dirs_exist_ok=True)
            
def check_if_merged(transcription_data, gap_threshold, start_word_cooldown):
    last_end_conversation_time = None
    last_start_conversation_time = None
    merged_details = []

    for segment in transcription_data:
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
                        merged_details.append((segment_start, start_word))
                        last_start_conversation_time = segment_start
                        break

    return bool(merged_details), merged_details

def split_merged_file(file_path, cutting_times, dest_folder_path):

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

def create_audio_database(audio_files):
    """
    Create a database of audio files with metadata and transcription data.

    Args:
    folder_path (str): Path to the folder containing audio files and their transcription data.

    Returns:
    pandas.DataFrame: A DataFrame containing metadata and transcription data for each audio file.
    """
    data = []

    for audio_file in audio_files:
        try:
            audio = MP3(audio_file)
            title = audio['TIT2'][0] if 'TIT2' in audio else 'Unknown'
            start_timestamp, end_timestamp = parse_timestamp_from_title(title)
            audio_length = audio.info.length

            transcription_path = audio_file.with_stem(audio_file.stem + '_transcription').with_suffix('.txt')
            transcription_data_path = audio_file.with_name(audio_file.stem + '_segments_data.json')
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
                "File Name": audio_file.stem,
                "File Path": str(audio_file),
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
        except Exception as e:
            print(f"Error processing file {audio_file}: {e}")

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

    def time_diff_okay(current_end, next_start):
        return 0 <= (next_start - current_end).total_seconds() <= time_delta

    for index, row in df_sorted.iterrows():
        file_path = row['File Path']
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
            # Other files - add to the current group if within time delta
            if current_group:
                last_index = df_sorted.index[df_sorted['File Path'] == current_group[-1]].tolist()[0]
                last_row = df_sorted.iloc[last_index]
                if time_diff_okay(last_row['End Timestamp'], row['Start Timestamp']):
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

def merge_associated_files(associated_files, df, dest_folder):
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

    # Delete old transcription folder
    for group in associated_files:
        for file_path in group:
            shutil.rmtree(Path(file_path).parent)

def main(args):
    # identify_and_split_merged_files(args.source_folder, args.dest_folder)

    # process_audio_files(args.dest_folder, args.dest_folder, move=True)

    audio_files = [audio_file for audio_file in Path(args.dest_folder).rglob('*.mp3') ]
    df_audio = create_audio_database(audio_files)
    associated_files = find_associated_files(df_audio)
    print(associated_files)
    merge_associated_files(associated_files, df_audio, args.dest_folder)

    process_audio_files(args.dest_folder, args.dest_folder, move=True)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)