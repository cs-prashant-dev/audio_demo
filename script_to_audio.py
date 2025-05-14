from TTS.api import TTS
from pydub import AudioSegment
import os

# Set up voices (choose from available models)
host_voice = "tts_models/en/vctk/vits"      # Example male/female voice
analyst_voice = "tts_models/en/ljspeech/tacotron2-DDC"  # Different voice

# Initialize TTS models
host_tts = TTS(model_name=host_voice)
analyst_tts = TTS(model_name=analyst_voice)

# Create output folder
output_dir = "output_audio"
os.makedirs(output_dir, exist_ok=True)

# Load transcript
with open("transcript.txt", "r") as file:
    lines = [line.strip() for line in file.readlines() if line.strip()]

audio_segments = []

# Generate audio per line
for idx, line in enumerate(lines):
    if line.startswith("Host:"):
        text = line.replace("Host:", "").strip()
        audio_path = os.path.join(output_dir, f"part_{idx}_host.wav")
        host_tts.tts_to_file(text=text, file_path=audio_path)
    elif line.startswith("Analyst:"):
        text = line.replace("Analyst:", "").strip()
        audio_path = os.path.join(output_dir, f"part_{idx}_analyst.wav")
        analyst_tts.tts_to_file(text=text, file_path=audio_path)
    else:
        continue

    audio_segments.append(audio_path)

# Combine all audio files
combined = AudioSegment.empty()
for path in audio_segments:
    segment = AudioSegment.from_file(path)
    combined += segment + AudioSegment.silent(duration=500)  # 0.5 sec pause

# Export final audio
combined.export("final_dialogue.mp3", format="mp3")

print("✅ Audio generation complete: final_dialogue.mp3")
