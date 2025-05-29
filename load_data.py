import numpy as np
import pandas as pd
from tqdm import tqdm 
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
import os
import gc

print("Wczytywanie danych...")

files_dirs = [
    ("csv/american.csv", "spectrograms_new/american_spec"),
    ("csv/american2.csv", "spectrograms_new/american2_spec"),
    ("csv/black.csv", "spectrograms_new/black_spec"),
    ("csv/butter.csv", "spectrograms_new/butter_spec"),
    ("csv/cassete.csv", "spectrograms_new/cassete_spec"),
    ("csv/crucial.csv", "spectrograms_new/crucial_spec"),
    ("csv/crucial2.csv", "spectrograms_new/crucial2_spec"),
    ("csv/desmond.csv", "spectrograms_new/desmond_spec"),
    ("csv/DX.csv", "spectrograms_new/DX_spec"),
    ("csv/flkeys.csv", "spectrograms_new/flkeys_spec"),
    ("csv/german.csv", "spectrograms_new/german_spec"),
    ("csv/japanese.csv", "spectrograms_new/japanese_spec"),
    ("csv/montuno.csv", "spectrograms_new/montuno_spec"),
    ("csv/neglected.csv", "spectrograms_new/neglected_spec"),
    ("csv/neglected2.csv", "spectrograms_new/neglected2_spec"),
    ("csv/notte.csv", "spectrograms_new/notte_spec"),
    ("csv/one.csv", "spectrograms_new/one_spec"),
    ("csv/ruffian.csv", "spectrograms_new/ruffian_spec"),
    ("csv/ruffian2.csv", "spectrograms_new/ruffian2_spec"),
    ("csv/savory.csv", "spectrograms_new/savory_spec"),
    ("csv/school.csv", "spectrograms_new/school_spec"),
    ("csv/strings.csv", "spectrograms_new/strings_spec"),
    ("csv/TBE.csv", "spectrograms_new/TBE_spec"),
    ("csv/tuck.csv", "spectrograms_new/tuck_spec"),
    ("csv/wunder.csv", "spectrograms_new/wunder_spec"),
    ("csv/savory2.csv", "spectrograms_new/savory2_spec"),
    ("csv/1982.csv", "spectrograms_new/1982_spec"),
    ("csv/german2.csv", "spectrograms_new/german2_spec"),
    ("csv/noir.csv", "spectrograms_new/noir_spec"),
    ("csv/butter2.csv", "spectrograms_new/butter2_spec"),
    ("csv/japanese2.csv", "spectrograms_new/japanese2_spec"),
    ("csv/savory3.csv", "spectrograms_new/savory3_spec"),
    ("csv/notte2.csv", "spectrograms_new/notte2_spec"),
    ("csv/grand.csv", "spectrograms_new/grand_spec"),
    ("csv/german3.csv", "spectrograms_new/german3_spec")
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

def load_and_stack_in_batches(file_paths, batch_size=2000):
    sample = np.load(file_paths[0])
    dtype = sample.dtype
    shape = sample.shape
    batched_data = []
    
    for i in tqdm(range(0, len(file_paths), batch_size), desc="Przetwarzanie partiami"):
        batch_paths = file_paths[i:i + batch_size]
        with ThreadPoolExecutor() as executor:
            batch = list(executor.map(np.load, batch_paths))
        temp_file = f"tmp/batch_{i//batch_size}.npy"
        np.save(temp_file, np.stack(batch))
        batched_data.append(temp_file)
    return batched_data, shape, dtype

os.makedirs("tmp", exist_ok=True)
batched_files, spec_shape, spec_dtype = load_and_stack_in_batches(file_paths)

def combine_batches(batched_files, output_path, shape, dtype):
    total_samples = sum(np.load(f, mmap_mode='r').shape[0] for f in batched_files)
    final_shape = (total_samples, *shape)
    X_combined = np.memmap(output_path, dtype=dtype, mode='w+', shape=final_shape)
    idx = 0
    for batch_file in tqdm(batched_files, desc="Łączenie partii"):
        batch = np.load(batch_file, mmap_mode='r')
        X_combined[idx:idx + len(batch)] = batch
        idx += len(batch)
        del batch
        gc.collect()
        os.remove(batch_file)
    return X_combined

X = combine_batches(batched_files, "tmp/X_memmap.npy", spec_shape, spec_dtype)

print("Liczba spektrogramów:", X.shape[0])

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