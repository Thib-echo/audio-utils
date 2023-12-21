from FasterWhisperTranscription import FasterWhisperTranscription
from concurrent.futures import ProcessPoolExecutor, as_completed

from pathlib import Path
from mutagen.mp3 import MP3
import json
import time
import shutil
import argparse
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(process)d] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def transcribe_and_time(transcription_service, audio_duration):
    start_time = time.time()
    transcription, segments_data = transcription_service.transcribe()
    end_time = time.time()

    transcription_time = end_time - start_time
    percentage = (transcription_time / audio_duration) * 100

    logging.info(f"{transcription_service.__class__.__name__}-{transcription_service.model_size} Transcription Time: {transcription_time:.2f} seconds ({percentage:.2f}% of audio duration)")
    
    return transcription, segments_data, transcription_time

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

def process_single_file(audio_path, new_folder, move, model_size):
    logging.info(f"Processing {audio_path}")

    # Initialize the model for the first time in each process
    if FasterWhisperTranscription.model is None:
        FasterWhisperTranscription.load_model(model_size)


    new_folder_path = new_folder / audio_path.stem

    # Check if output folder is already created and its sanity
    transcript_file_path = new_folder_path / (audio_path.stem + "_transcription.txt")
    transcript_json_path = new_folder_path / (audio_path.stem + "_segments_data.json")
    if new_folder_path.exists() and all([
        new_folder_path.joinpath(audio_path.name).exists(),
        transcript_file_path.exists(),
        transcript_json_path.exists()
    ]):
        logging.info("Folder already created and sane, no processing needed, continuing ...")
        return audio_path, 0

    # If the folder exists but is not sane, delete it
    if new_folder_path.exists():
        logging.info("Folder already created but corrupted, deletion and start a new transcription ...")
        shutil.rmtree(new_folder_path)

    audio_duration = get_audio_duration_from_metadata(audio_path)

    # Transcribe the audio file
    faster_whisper = FasterWhisperTranscription(str(audio_path), model_size)
    transcription_fw, segments_data, _ = transcribe_and_time(faster_whisper, audio_duration)

    # Create a new folder for the audio and its transcript
    new_folder_path.mkdir(parents=True)

    # Move the audio file to the new folder
    destination = str(new_folder_path / audio_path.name)
    if move:
        shutil.move(str(audio_path), destination)
    else:
        shutil.copy(str(audio_path), destination)

    # Write the transcript to a text file in the new folder
    with open(transcript_file_path, "w", encoding="utf-8") as transcript_file:
        transcript_file.write(transcription_fw)

    with open(transcript_json_path, "w", encoding="utf-8") as transcript_json:
        json.dump(segments_data, transcript_json, indent=4, ensure_ascii=False)

    logging.info(f"Processed and moved {audio_path.name}")
    return audio_path, audio_duration

def process_audio_files(folder_path, new_folder, move, model_size, max_workers=4):
    folder_path = Path(folder_path)
    new_folder = Path(new_folder)
    total_audio_duration = 0

    if max_workers is None:
        max_workers = os.cpu_count()  # Or set a default value

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_file, audio_path, new_folder, move, model_size)
                   for audio_path in folder_path.glob('*.mp3')]

        for future in as_completed(futures):
            _, audio_duration = future.result()
            total_audio_duration += audio_duration

    return total_audio_duration


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process audio files in a specified folder.")
    parser.add_argument("--source_folder", type=str, help="Path to the folder containing audio files")
    parser.add_argument("--dest_folder", type=str, help="Path to the folder where audio files will be saved along their transcription")
    parser.add_argument("--model_size", type=str, default="large-v3", help="Whisper model size (tiny, base, medium, large, large-v2, large-v3)")
    parser.add_argument("--move", action="store_true", default=False, help="Move original mp3 file instead of copying it")
    args = parser.parse_args()

    start_time = time.time()
    total_audio_duration = process_audio_files(args.source_folder, args.dest_folder, args.move, args.model_size)
    total_end_time = time.time()
    total_running_time = total_end_time - start_time
    percentage = (total_running_time / total_audio_duration) * 100

    logging.info(f"Total Running Time: {total_running_time:.2f} seconds or {percentage:.2f}% of total audio duration")