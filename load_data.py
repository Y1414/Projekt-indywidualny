import numpy as np
import pandas as pd
from tqdm import tqdm 
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
import os

print("Wczytywanie danych...")

files_dirs = [
    ("csv/american.csv", "spectrograms/american_spec"),
    ("csv/american2.csv", "spectrograms/american2_spec"),
    ("csv/black.csv", "spectrograms/black_spec"),
    ("csv/butter.csv", "spectrograms/butter_spec"),
    ("csv/cassete.csv", "spectrograms/cassete_spec"),
    ("csv/crucial.csv", "spectrograms/crucial_spec"),
    ("csv/crucial2.csv", "spectrograms/crucial2_spec"),
    ("csv/desmond.csv", "spectrograms/desmond_spec"),
    ("csv/DX.csv", "spectrograms/DX_spec"),
    ("csv/flkeys.csv", "spectrograms/flkeys_spec"),
    ("csv/german.csv", "spectrograms/german_spec"),
    ("csv/japanese.csv", "spectrograms/japanese_spec"),
    ("csv/montuno.csv", "spectrograms/montuno_spec"),
    ("csv/neglected.csv", "spectrograms/neglected_spec"),
    ("csv/neglected2.csv", "spectrograms/neglected2_spec"),
    ("csv/notte.csv", "spectrograms/notte_spec"),
    ("csv/one.csv", "spectrograms/one_spec"),
    ("csv/ruffian.csv", "spectrograms/ruffian_spec"),
    ("csv/ruffian2.csv", "spectrograms/ruffian2_spec"),
    ("csv/savory.csv", "spectrograms/savory_spec"),
    ("csv/school.csv", "spectrograms/school_spec"),
    ("csv/strings.csv", "spectrograms/strings_spec"),
    ("csv/TBE.csv", "spectrograms/TBE_spec"),
    ("csv/tuck.csv", "spectrograms/tuck_spec"),
    ("csv/wunder.csv", "spectrograms/wunder_spec"),
    ("csv/savory2.csv", "spectrograms/savory2_spec"),
    ("csv/1982.csv", "spectrograms/1982_spec"),
    ("csv/german2.csv", "spectrograms/german2_spec"),
    ("csv/noir.csv", "spectrograms/noir_spec"),
    ("csv/butter2.csv", "spectrograms/butter2_spec"),
    ("csv/japanese2.csv", "spectrograms/japanese2_spec"),
    ("csv/savory3.csv", "spectrograms/savory3_spec"),
    ("csv/notte2.csv", "spectrograms/notte2_spec"),
    ("csv/grand.csv", "spectrograms/grand_spec"),
    ("csv/german3.csv", "spectrograms/german3_spec")
]

print("Liczba plików CSV:", len(files_dirs))

def load_and_process(file_path, directory):
    df = pd.read_csv(file_path, engine='c') 
    df["directory"] = directory  
    return df

with ThreadPoolExecutor() as executor:
    futures = [executor.submit(load_and_process, file, dir) for file, dir in files_dirs]
    dfs = [future.result() for future in futures]
    df = pd.concat(dfs, ignore_index=True)


print("Liczba próbek:", len(df))
file_paths = [os.path.join(row["directory"], row["file"]) for _, row in df.iterrows()]

def load_spectrogram(path):
    return np.load(path)

with ThreadPoolExecutor() as executor:
    X = list(tqdm(executor.map(load_spectrogram, file_paths), total=len(file_paths), desc="Wczytywanie spektrogramów"))

X = np.stack(X)  

print("Liczba spektrogramów:", len(X))

y = np.array(df.iloc[:, 1:-1])  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_min = np.min(X_train, axis=(0, 1, 2), keepdims=True)
X_max = np.max(X_train, axis=(0, 1, 2), keepdims=True)

def normalize_in_batches(data, X_min, X_max, batch_size=200000):
    for i in range(0, len(data), batch_size):
        data[i:i+batch_size] = (data[i:i+batch_size] - X_min) / (X_max - X_min + 1e-8)
        print(f"Processed batch {i//batch_size + 1}/{(len(data)-1)//batch_size + 1}")
    return data

X_train = normalize_in_batches(X_train, X_min, X_max)
X_test = normalize_in_batches(X_test, X_min, X_max)

X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

np.save("tmp/X_train.npy", X_train)
np.save("tmp/y_train.npy", y_train)
np.save("tmp/X_test.npy", X_test)
np.save("tmp/y_test.npy", y_test)

print("Liczba próbek:", len(X))
print("Rozmiar X:", X.shape)
print("Rozmiar y:", y.shape)

print("Czy są NaN w X_train?", np.isnan(X_train).sum())
print("Czy są Inf w X_train?", np.isinf(X_train).sum())
print("Czy są NaN w y_train?", np.isnan(y_train).sum())
print("Czy są Inf w y_train?", np.isinf(y_train).sum())

print("Min X:", np.min(X_train), "Max X:", np.max(X_train))
print("Min y:", np.min(y_train), "Max y:", np.max(y_train))