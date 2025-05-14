import whisper
from pyannote.audio import Pipeline
from pyannote.core import Segment
import os

# ----------------------------------
# SETUP
# ----------------------------------

# 1. Whisper model
whisper_model = whisper.load_model("large")  # or "base", "medium", etc.

# 2. Path to audio file (must be WAV, mono, 16kHz)
audio_file = "your_podcast.wav"

# 3. Your Hugging Face token
HUGGINGFACE_TOKEN = ""  # replace this with your token

# 4. Load pyannote speaker diarization pipeline
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization",
    use_auth_token=HUGGINGFACE_TOKEN
)

# ----------------------------------
# STEP 1: Transcribe with Whisper
# ----------------------------------

print("🔤 Running Whisper transcription...")
transcription = whisper_model.transcribe(audio_file, language="en")

# ----------------------------------
# STEP 2: Run diarization
# ----------------------------------

print("🧠 Running speaker diarization...")
diarization = diarization_pipeline(audio_file)

# ----------------------------------
# STEP 3: Combine transcription with speaker labels
# ----------------------------------

def label_transcript_by_speaker(segments, diarization):
    labeled_segments = []
    for segment in segments:
        start = segment['start']
        end = segment['end']
        text = segment['text'].strip()

        # Find matching speaker label from diarization
        matched_speaker = "Unknown"
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if turn.start <= end and turn.end >= start:
                matched_speaker = speaker
                break

        labeled_segments.append({
            "speaker": matched_speaker,
            "start": start,
            "end": end,
            "text": text
        })
    return labeled_segments

final_transcript = label_transcript_by_speaker(transcription['segments'], diarization)

# ----------------------------------
# STEP 4: Display or save result
# ----------------------------------

for seg in final_transcript:
    print(f"{seg['speaker']} ({seg['start']:.2f} - {seg['end']:.2f}): {seg['text']}")

# Optional: Save to text file
with open("final_transcript.txt", "w") as f:
    for seg in final_transcript:
        f.write(f"{seg['speaker']} ({seg['start']:.2f} - {seg['end']:.2f}): {seg['text']}\n")
