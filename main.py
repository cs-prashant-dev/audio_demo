import whisper
from resemblyzer import VoiceEncoder, preprocess_wav, sampling_rate
from sklearn.cluster import KMeans
import numpy as np
import librosa

AUDIO_FILE = "your_podcast.wav"  # Use 16kHz mono .wav
NUM_SPEAKERS = 2

# Step 1: Transcribe
print("🔤 Transcribing with Whisper...")
model = whisper.load_model("base")
result = model.transcribe(AUDIO_FILE)
segments = result['segments']

# Step 2: Speaker Diarization
print("🧠 Running speaker diarization...")
wav, _ = librosa.load(AUDIO_FILE, sr=sampling_rate)
wav = preprocess_wav(wav)
encoder = VoiceEncoder()
_, partial_embeds, wav_splits = encoder.embed_utterance(wav, return_partials=True, rate=16)

# Convert slice frames to seconds
frame_duration = 1 / sampling_rate
split_times = [slice_.start * frame_duration for slice_ in wav_splits]

# Step 3: Cluster speakers
kmeans = KMeans(n_clusters=NUM_SPEAKERS).fit(partial_embeds)
labels = kmeans.labels_

import soundfile as sf

for speaker_id in range(NUM_SPEAKERS):
    try:
        idx = list(labels).index(speaker_id)
        start_sec = split_times[idx]
        duration_sec = 2.0  # extract 2 seconds
        sample, sr = librosa.load(AUDIO_FILE, sr=None, offset=start_sec, duration=duration_sec)
        sf.write(f"speaker_{speaker_id}.wav", sample, sr)
        print(f"✅ Saved example for Speaker {speaker_id} as 'speaker_{speaker_id}.wav'")
    except ValueError:
        print(f"⚠️ Speaker {speaker_id} not found in labels.")

# Step 4: Assign speaker ID
def find_speaker(start_time):
    idx = np.argmin([abs(t - start_time) for t in split_times])
    return labels[idx]

# Step 5: Print speaker-labeled transcript
print("\n📝 Speaker-Labeled Transcript:\n")
with open("transcript_speakers.txt", "w", encoding="utf-8") as f:
    for seg in segments:
        speaker_id = find_speaker(seg['start'])
        line = f"[Speaker {speaker_id}] [{seg['start']:.1f}-{seg['end']:.1f}]: {seg['text'].strip()}"
        print(line)
        f.write(line + "\n")

print("\n✅ Transcript saved as 'transcript_speakers.txt'")
