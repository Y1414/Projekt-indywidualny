import sounddevice as sd
import numpy as np
import librosa
import time
import tensorflow.keras as keras
import io
from pydub import AudioSegment
import tensorflow_addons as tfa
import tensorflow as tf
from tensorflow_addons.losses import SigmoidFocalCrossEntropy
from model import focal_loss_custom
from generate_spec import generate_mel_spectrogram

# Wczytaj model
model = keras.models.load_model(
    "models/piano_note_model_2048_v0.5.keras",
    custom_objects={'custom_loss': focal_loss_custom}
)

# Parametry
sr = 44100
segment_duration = 0.05  # 50 ms
samples_per_segment = int(sr * segment_duration)

# Funkcja do generowania spektrogramu z numpy array

# Główna pętla nasłuchująca
print("Nasłuchiwanie... Naciśnij Ctrl+C, aby przerwać.")


try:
    while True:
        audio = sd.rec(samples_per_segment, samplerate=sr, channels=1, dtype='int16')
        sd.wait()

        y = audio.astype(np.float32).flatten() / 32768.0

        # Generowanie spektrogramu
        spec = generate_mel_spectrogram(y, sr=sr)
        spec_min = np.min(spec)
        spec_max = np.max(spec)
        spec = (spec - spec_min) / (spec_max - spec_min + 1e-8)
        spec = np.expand_dims(spec, axis=-1)  # Kanał
        spec = np.expand_dims(spec, axis=0)   # Batch

        prediction = model.predict(spec, verbose=0)[0]
        if prediction.max() > 0.2:
            threshold = prediction.max() * 0.4
        else:
            threshold = 100

        expected = []
        pressed_keys = []
        for i, val in enumerate(prediction):
            if val > threshold:
                pressed_keys.append((i, val))


        print(f"Nuty: {pressed_keys}", )

        # Czekaj tylko tyle, ile potrzeba (np. około 50 ms)
        time.sleep(0.01)

except KeyboardInterrupt:
    print("Zatrzymano.")
