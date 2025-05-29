import librosa
import numpy as np
import os
from tqdm import tqdm
from librosa.util import fix_length

def generate_mel_spectrogram(y, sr=44100, n_mels=109, hop_length=128, n_fft=1024):
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        hop_length=hop_length,
        n_fft=n_fft,
        fmin=27.5, 
        fmax=4200, 
        center=False
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db
def process_audio_in_chunks_librosa(audio_file, segment_length_ms, output_folder, sr=44100):
    y, _ = librosa.load(audio_file, sr=sr, mono=True)
    segment_length = int(sr * segment_length_ms / 1000)

    os.makedirs(output_folder, exist_ok=True)

    mel_spec_count = 0
    for start in range(0, len(y), segment_length):
        segment = y[start:start + segment_length]
        if len(segment) < segment_length:
            break 

        segment = fix_length(segment, size=segment_length)  
        mel_spec = generate_mel_spectrogram(segment, sr)
        np.save(os.path.join(output_folder, f'mel_spec_{mel_spec_count}.npy'), mel_spec)
        mel_spec_count += 1

if __name__ == "__main__":
    audio_folder = 'audio'
    names = [os.path.splitext(file)[0] for file in os.listdir(audio_folder) if file.endswith('.wav')]

    for name in tqdm(names, desc="Przetwarzanie plików audio"):
        audio_file = f'audio/{name}.wav'
        output_folder = f'spectrograms_new/{name}_spec'
        segment_length_ms = 50  

        process_audio_in_chunks_librosa(audio_file, segment_length_ms, output_folder)
