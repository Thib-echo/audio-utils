from dotenv import load_dotenv
import torch
import soundfile as sf
import whisper
from pyannote.audio import Pipeline
from pydub import AudioSegment
from pathlib import Path
from os import environ as env
import time

load_dotenv()

class PyannoteDiarizer:
    def __init__(self, audio_path, nb_speaker):
        self.audio_path = audio_path
        self.nb_speaker = nb_speaker
        self.hf_token = env["HF_TOKEN"]

    @staticmethod
    def get_torch_device():
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def diarize(self):
        try:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.0", use_auth_token=self.hf_token
            )
            pipeline.to(self.get_torch_device())

            diarization = pipeline(self.audio_path, num_speakers=self.nb_speaker)
        except Exception as e:
            print(f"Pyannote Diarization failed: {e}")
            return None

        return self._format_diarization_result(diarization)

    def _format_diarization_result(self, diarization):
        speaker_segments = {}
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if speaker not in speaker_segments:
                speaker_segments[speaker] = []
            speaker_segments[speaker].append((turn.start, turn.end))

        return speaker_segments

    def transcribe(self):
        model = whisper.load_model("base", device=self.get_torch_device())
        audio = whisper.load_audio(self.audio_path)
        result = model.transcribe(audio)
        return result

    def combine_diarization_and_transcription(self, diarization, transcription):
        combined_result = {}
        for segment in transcription['segments']:
            for speaker, times in diarization.items():
                for start, end in times:
                    if start <= segment['start'] and end >= segment['end']:
                        if speaker not in combined_result:
                            combined_result[speaker] = []
                        combined_result[speaker].append(segment['text'])
                        break
        return combined_result


# Usage
start = time.time()

audio_path = Path("./audio_regrouping/group_1/2023_1_15_10_5_41_ch30_merged.mp3")

sound = AudioSegment.from_mp3(audio_path)
wav_path = audio_path.with_suffix(".wav")
sound.export(wav_path, format="wav")

diarizer = PyannoteDiarizer(wav_path, nb_speaker=2)

diarization = diarizer.diarize()
transcription = diarizer.transcribe()
combined_result = diarizer.combine_diarization_and_transcription(diarization, transcription)
end = time.time()
print(f"Elapsed time: {end - start}")

for speaker, text in combined_result.items():
    print(f"Speaker {speaker}: {' '.join(text)}")
