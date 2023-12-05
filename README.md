# Audio Utility Project

## Overview
The aim of this project is to group together utility functions for processing audio files.


## Processing available
- Transcribe all audio file in a folder and save it in an other folder
- Audio fusion script based on timestamp and word checking ("boujour", "au revoir", or both)

## Installation

### Prerequisites
- Python 3.11 or higher
- Pip package manager

### Setting Up
Clone the repository or download the source code:

```bash
git clone git@github.com:Thib-echo/audio-utils.git
cd audio-utils
```
Create a virtual env and activate it:
```bash
python -m venv env
source env/bin/activate # For linux users
.\env\Script\activate # For windows users
```

### Dependencies
Install the necessary Python packages using pip:

```bash
pip install -r requirements.txt
```

## Usage
In order to get the transcriptions for all audio in a folder, use this command :

```bash
python .\transcribe_all.py --source_folder "path/to/folder/with/audios/to/transcribe/" --dest_folder "path/to/saving/folder/" 
```
This will save the audio in the other folder in this form :

```
saving_folder
-- audio_1
---- audio_1.mp3 
---- audio_1_transcription.txt
-- audio_2
---- audio_2.mp3 
---- audio_2_transcription.txt
```