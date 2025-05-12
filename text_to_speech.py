from TTS.api import TTS
import numpy as np
import soundfile as sf
import re
import os

# Load multi-speaker TTS model
tts = TTS(model_name="tts_models/multispeaker/en/vctk/vits")

# # Get available speaker names
# print(tts.speakers)

# Map Speaker ID to Coqui voice names
speaker_voice_map = {
    "0": "p335",  # Male
    "1": "p243",  # Female
}

# File paths
input_file = "transcript_speakers.txt"
output_folder = "segments"
os.makedirs(output_folder, exist_ok=True)

# Read transcript and synthesize speech
segment_paths = []
with open(input_file, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f.readlines()):
        match = re.match(r"\[Speaker (\d)\] \[(\d+\.\d+)-(\d+\.\d+)\]: (.+)", line.strip())
        if not match:
            print(f"❌ Skipping invalid line: {line}")
            continue

        speaker_id, start, end, text = match.groups()
        voice = speaker_voice_map.get(speaker_id, "p335")  # Default to male if unknown

        output_path = os.path.join(output_folder, f"segment_{idx}.wav")
        print(f"🔊 Generating Segment {idx} | Speaker {speaker_id} ({voice}) | Text: {text}")
        tts.tts_to_file(text=text, speaker=voice, file_path=output_path)
        segment_paths.append(output_path)

# Merge all segments into one podcast
print("\n🔗 Merging segments into final_podcast.wav...")
final_audio = []
for path in segment_paths:
    data, sr = sf.read(path)
    final_audio.append(data)

combined = np.concatenate(final_audio)
sf.write("final_podcast.wav", combined, sr)
print("✅ Podcast saved as final_podcast.wav")
