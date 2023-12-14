from faster_whisper import WhisperModel

class FasterWhisperTranscription:
    model = None

    @classmethod
    def load_model(cls, model_size):
        if cls.model is None:
            cls.model = WhisperModel(model_size, device="cuda", compute_type="int8")

    def __init__(self, audio_path, model_size):
        self.audio_path = audio_path
        self.model_size = model_size
        FasterWhisperTranscription.load_model(model_size)

    @staticmethod
    def format_faster_whisper_output(faster_whisper_output):
        # Concatenate all transcriptions into one string
        segments_data = []

        for segment in faster_whisper_output:
            segment_data = {
                "id": segment.id,
                "seek": segment.seek,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "tokens": segment.tokens,
                "temperature": segment.temperature,
                "avg_logprob": segment.avg_logprob,
                "compression_ratio": segment.compression_ratio,
                "no_speech_prob": segment.no_speech_prob,
                "words": [
                    {
                        "start": word.start,
                        "end": word.end,
                        "word": word.word,
                        "probability": word.probability
                    } for word in segment.words
                ]
            }
            segments_data.append(segment_data)

        concatenated_transcription = ' '.join(segment['text'] for segment in segments_data)
        return concatenated_transcription, segments_data

    def transcribe(self):
        print(f"Starting transcription for FasterWhisper-{self.model_size}")
        segments, _ = self.model.transcribe(self.audio_path, beam_size=5, vad_filter=True, word_timestamps=True)
        return self.format_faster_whisper_output(segments)
