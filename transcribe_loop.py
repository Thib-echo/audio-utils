import subprocess
import time
import argparse
from pathlib import Path

def count_files_with_extension(parent_directory, extension):
    """
    Counts the number of files with a given extension in the specified directory
    and its subdirectories using pathlib's rglob.
    """
    parent_path = Path(parent_directory)
    return sum(1 for file in parent_path.rglob('*' + extension))

def is_transcription_complete(source_folder, dest_folder):
    """
    Checks if the number of transcription files in the subfolders of the destination folder
    matches the number of audio files in the source folder using pathlib's rglob.
    """
    num_audio_files = count_files_with_extension(source_folder, '.mp3')
    num_transcription_files = count_files_with_extension(dest_folder, '_transcription.txt')

    return num_audio_files == num_transcription_files

def run_transcription_script(source_folder, dest_folder, move):
    """
    Runs the transcription script and restarts it if it stops unexpectedly.
    """

    venv_python = Path("env/Scripts/python.exe") # Use 'venv/Scripts/python.exe' on Windows

    script_command = [
        str(venv_python), "./transcribe_all.py",
        "--source_folder", source_folder,
        "--dest_folder", dest_folder
    ]
    if move:
        script_command.append("--move")

    while not is_transcription_complete(source_folder, dest_folder):
        print("Starting transcription script...")
        process = subprocess.Popen(script_command)
        process.wait()  # Wait for the script to complete or fail

        if process.returncode != 0:
            print(f"Script stopped unexpectedly with return code {process.returncode}. Restarting...")
            time.sleep(5)  # Wait for 5 seconds before restarting
        else:
            print("Script completed successfully.")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wrapper to monitor and restart the transcription script.")
    parser.add_argument("--source_folder", type=str, help="Path to the folder containing audio files")
    parser.add_argument("--dest_folder", type=str, help="Path to the folder where audio files will be saved along their transcription")
    parser.add_argument("--move", action="store_true", default=False, help="Move original mp3 file instead of copying it")
    args = parser.parse_args()

    run_transcription_script(args.source_folder, args.dest_folder, args.move)
