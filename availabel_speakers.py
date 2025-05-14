from TTS.api import TTS
from pydub import AudioSegment
import os

# Load a multi-speaker TTS model
tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts")

# List speakers
speakers = tts.speakers
print(f"Total speakers found: {len(speakers)}")

# Output folder
os.makedirs("sample_speakers", exist_ok=True)

# Sample sentence
sample_text = "This is a test voice. I am a sample speaker from Coqui TTS."

# Generate sample for each speaker
for speaker in speakers:
    file_path = f"sample_speakers/{speaker}.wav"
    print(f"Generating sample for: {speaker}")
    tts.tts_to_file(
    text=sample_text,
    speaker=speaker,
    language="en",   # Specify the language
    file_path=file_path
)

print("\n✅ All speaker samples saved in 'sample_speakers/' folder.")
