from FasterWhisperTranscription import FasterWhisperTranscription
from pathlib import Path
from mutagen.mp3 import MP3
import json
import time
import shutil
import argparse
import subprocess
from multiprocessing import Pool


def transcribe_and_time(transcription_service, audio_duration):
    start_time = time.time()
    transcription, segments_data = transcription_service.transcribe()
    end_time = time.time()

    transcription_time = end_time - start_time
    percentage = (transcription_time / audio_duration) * 100

    print(f"{transcription_service.__class__.__name__}-{transcription_service.model_size} Transcription Time: {transcription_time:.2f} seconds ({percentage:.2f}% of audio duration)")
    
    return transcription, segments_data, transcription_time

def get_audio_duration_from_metadata(audio_path):
    audio_file = Path(audio_path)

    if not audio_file.exists():
        raise FileNotFoundError(f"File not found: {audio_path}")

    # Check file extension and use appropriate Mutagen class
    file_extension = audio_file.suffix.lower()
    if file_extension == '.mp3':
        audio = MP3(audio_file)  # Open the MP3 file
        duration = audio.info.length
        # No explicit close method, but we can delete the object to free up resources
        del audio
    else:
        raise NotImplementedError(f"Metadata retrieval for {file_extension} files is not implemented.")

    return duration

def debug_audio_files(file_path):
    folder_path = file_path.parent
    file = file_path.name
    output_file = folder_path / f"processed_{file}"

    command = f'ffmpeg -i "{file_path}" -vn -ar 16000 -c:a libmp3lame -y "{output_file}"'

    try:
        subprocess.run(command, capture_output=True, check=True, text=True)
        file_path.unlink()  # Remove the original file
        output_file.rename(file_path)  # Rename the processed file to the original file's name
        print(f"ffmpeg runned on: {file}")
    except subprocess.CalledProcessError as e:
        print(f"Error processing {file}: {e.stderr}")

def process_audio_files(folder_path, new_folder, move):
    folder_path = Path(folder_path)
    new_folder = Path(new_folder)
    total_audio_duration = 0

    audio_files = list(folder_path.glob('*.mp3'))

    with Pool() as pool:
        pool.map(debug_audio_files, audio_files)


    for audio_path in folder_path.glob('*.mp3'):  # Repeat for other formats as needed
        print(f"Processing {audio_path}")

        new_folder_path = new_folder / audio_path.stem

        # Check if output folder is already created and its sanity
        transcript_file_path = new_folder_path / (audio_path.stem + "_transcription.txt")
        transcript_json_path = new_folder_path / (audio_path.stem + "_segments_data.json")
        if new_folder_path.exists() and all([
            new_folder_path.joinpath(audio_path.name).exists(),
            transcript_file_path.exists(),
            transcript_json_path.exists()
        ]):
            print("Folder already created and sane, no processing needed, continuing ...")
            continue



        # If the folder exists but is not sane, delete it
        if new_folder_path.exists():
            # print("Folder already created but corrupted, deletion and start a new transcription ...")
            # shutil.rmtree(new_folder_path)

            # Create a new folder for the audio and its transcript
            continue

        new_folder_path.mkdir(parents=True)
        
        audio_duration = get_audio_duration_from_metadata(audio_path)
        total_audio_duration += audio_duration

        # Transcribe the audio file
        faster_whisper = FasterWhisperTranscription(str(audio_path), "large-v3")
        transcription_fw, segments_data, _ = transcribe_and_time(faster_whisper, audio_duration)

        # Move the audio file to the new folder
        if move:
            shutil.move(str(audio_path), str(new_folder_path / audio_path.name))
        else:
            shutil.copy(str(audio_path), str(new_folder_path / audio_path.name))

        # Write the transcript to a text file in the new folder
        with open(transcript_file_path, "w", encoding="utf-8") as transcript_file:
            transcript_file.write(transcription_fw)

        with open(transcript_json_path, "w", encoding="utf-8") as transcript_json:
            json.dump(segments_data, transcript_json, indent=4, ensure_ascii=False)

        print(f"Processed and moved {audio_path.name}")

    return total_audio_duration


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process audio files in a specified folder.")
    parser.add_argument("--source_folder", type=str, help="Path to the folder containing audio files")
    parser.add_argument("--dest_folder", type=str, help="Path to the folder where audio files will be saved along their transcription")
    parser.add_argument("--move", action="store_true", default=False, help="Move original mp3 file instead of copying it")
    args = parser.parse_args()

    start_time = time.time()
    total_audio_duration = process_audio_files(args.source_folder, args.dest_folder, args.move)
    total_end_time = time.time()
    total_running_time = total_end_time - start_time
    percentage = (total_running_time / total_audio_duration) * 100

    print(f"Total Running Time: {total_running_time:.2f} seconds or {percentage:.2f}% of total audio duration")