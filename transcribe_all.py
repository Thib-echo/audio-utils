from FasterWhisperTranscription import FasterWhisperTranscription
from pathlib import Path
from mutagen.mp3 import MP3

import time
import shutil
import argparse

def transcribe_and_time(transcription_service, audio_duration):
    start_time = time.time()
    transcription = transcription_service.transcribe()
    end_time = time.time()

    transcription_time = end_time - start_time
    percentage = (transcription_time / audio_duration) * 100

    print(f"{transcription_service.__class__.__name__}-{transcription_service.model_size} Transcription Time: {transcription_time:.2f} seconds ({percentage:.2f}% of audio duration)")
    
    return transcription, transcription_time

def get_audio_duration_from_metadata(audio_path):
    audio_file = Path(audio_path)

    if not audio_file.exists():
        raise FileNotFoundError(f"File not found: {audio_path}")

    # Check file extension and use appropriate Mutagen class
    file_extension = audio_file.suffix.lower()
    if file_extension == '.mp3':
        audio = MP3(audio_file)
    else:
        raise NotImplementedError(f"Metadata retrieval for {file_extension} files is not implemented.")

    # Retrieve duration
    duration = audio.info.length
    return duration

def process_audio_files(folder_path, new_folder):
    print(folder_path, new_folder)
    folder_path = Path(folder_path)
    new_folder = Path(new_folder)
    total_audio_duration = 0

    for audio_path in folder_path.rglob('*.mp3'):  # Repeat for other formats as needed
        print(f"Processing {audio_path}")
        audio_duration = get_audio_duration_from_metadata(audio_path)
        total_audio_duration += audio_duration

        # Transcribe the audio file
        faster_whisper = FasterWhisperTranscription(str(audio_path), "large-v3")
        transcription_fw, _ = transcribe_and_time(faster_whisper, audio_duration)

        # Create a new folder for the audio and its transcript
        new_folder_path = new_folder / audio_path.stem
        new_folder_path.mkdir(exist_ok=True)

        # Move the audio file to the new folder
        shutil.move(str(audio_path), str(new_folder_path / audio_path.name))

        # Write the transcript to a text file in the new folder
        transcript_file_path = new_folder_path / (audio_path.stem + ".txt")
        with open(transcript_file_path, "w") as transcript_file:
            transcript_file.write(transcription_fw)

        print(f"Processed and moved {audio_path.name}")

    return total_audio_duration


# Parse command line arguments
parser = argparse.ArgumentParser(description="Process audio files in a specified folder.")
parser.add_argument("--source_folder", type=str, help="Path to the folder containing audio files")
parser.add_argument("--dest_folder", type=str, help="Path to the folder where audio files will be saved along their transcription")
args = parser.parse_args()

start_time = time.time()
total_audio_duration = process_audio_files(args.source_folder, args.dest_folder)
total_end_time = time.time()
total_running_time = total_end_time - start_time
percentage = (total_running_time / total_audio_duration) * 100

print(f"Total Running Time: {total_running_time:.2f} seconds or {percentage:.2f}% of total audio duration")