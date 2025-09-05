import librosa
from transformers import pipeline
classifier = pipeline(
    "audio-classification",
    model="superb/wav2vec2-base-superb-er"
)
file_path = "pussy.wav"
signal, sample_rate = librosa.load(file_path, sr=16000, mono=True)
prediction = classifier(signal, sampling_rate=sample_rate)
top_emotion = prediction[0]['label']
top_score = prediction[0]['score']
print(f"Predicted Emotion: {top_emotion} (Score: {top_score:.2f})")
