import librosa
import uvicorn
from transformers import pipeline
from fastapi import FastAPI, UploadFile, File
import io

app = FastAPI()

# Load the AI model once when the server starts.
classifier = pipeline(
    "audio-classification",
    model="superb/wav2vec2-base-superb-er"
)


#creating an api-endpoint
@app.post("/predict")
async def predict_emotion(audio_file: UploadFile = File(...)):
    # Read the audio file from the request
    audio_bytes = await audio_file.read()

    # Process the audio with librosa
    signal, sample_rate = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)

    #Get the prediction
    prediction = classifier(signal, sampling_rate=sample_rate)

    #picking the top result
    top_result = prediction[0]
    emotion = top_result['label']
    score = top_result['score']

    #returning the output
    return {
        "detected_emotion": emotion,
        "confidence_score": score
    }
