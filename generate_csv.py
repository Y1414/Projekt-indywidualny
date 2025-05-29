import os
import pandas as pd
import numpy as np

excluded = {"japanese2", "savory3", "notte2", "grand", "german3"}
audio_folder = 'labels'
names = [
    os.path.splitext(file)[0]
    for file in os.listdir(audio_folder)
    if file.endswith('.npy') and os.path.splitext(file)[0] not in excluded
]
for name in names:

    labels = np.load(f'labels/{name}.npy')

    double_labels = []

    for i in labels:
        for j in range(2):
            double_labels.append(i)

    spec_folder = f'spectrograms_new/{name}_spec'

    spec_files = sorted(os.listdir(spec_folder), key=lambda x: int(x.split("_")[-1].split(".")[0]))


    number_to_remove = 2

    for i in spec_files:
        if str(number_to_remove) in i:
            spec_files.remove(i)
            number_to_remove += 5

    print(spec_files[:20])
    print(len(double_labels), len(spec_files))

    assert len(double_labels) == len(spec_files)

    csv_data = []
    for file_name, label in zip(spec_files, double_labels):
        csv_data.append([file_name] + list(label))

    df = pd.DataFrame(csv_data, columns=["file"] + [f"k{i}" for i in range(88)])
    df.to_csv(f"csv/{name}.csv", index=False)

    print(f"Zapisano plik {name}.csv z {len(df)} wierszami.")