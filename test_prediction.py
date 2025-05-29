import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import os
import pandas as pd
import tensorflow_addons as tfa

model = keras.models.load_model("models/piano_note_model_commercial_v0.6.keras")

# X_test = np.load("tmp/X_test.npy", mmap_mode='r')
# y_test = np.load("tmp/y_test.npy", mmap_mode='r')

# test_results = model.evaluate(X_test, y_test)

# print(f"Test Loss: {test_results[0]:.4f}")
# print(f"Test Accuracy: {test_results[1]:.4f}")
# print(f"Test AUC: {test_results[2]:.4f}")

# data = pd.read_csv("sample.csv")

test_files = []
real_vectors = []

test_files = ["mel_spec_15.npy", "mel_spec_10.npy","mel_spec_5.npy","mel_spec_2.npy", "mel_spec_1.npy", "mel_spec_0.npy", "mel_spec_3.npy", "mel_spec_4.npy", "mel_spec_6.npy", "mel_spec_7.npy", "mel_spec_8.npy", "mel_spec_9.npy", "mel_spec_11.npy", "mel_spec_12.npy", "mel_spec_13.npy", "mel_spec_14.npy"]

number_of_files = len(test_files)

# vector = [1 if i in [29,34,41] else 0 for i in range(88)]
vector = [1 if i in [39, 43, 46] else 0 for i in range(88)]
print(vector)

real_vectors = []
for i in range(number_of_files):
    real_vectors.append(vector)

i=0

added_lost_vector = []

for file_name in test_files:
    spec = np.load(os.path.join("spectrograms/test3_spec", file_name))

    spec_min = np.min(spec)
    spec_max = np.max(spec)
    spec = (spec - spec_min) / (spec_max - spec_min + 1e-8)
    spec = np.expand_dims(spec, axis=-1)  
    spec = np.expand_dims(spec, axis=0)  

    prediction = model.predict(spec)

    threshold = prediction[0].max() * 0.7
    print(threshold)

    a_l = [0,0]

    predicted_indicies = []
    print("Predicted keys:")
    for j in range(len(prediction[0])):
        if (prediction[0][j] > threshold):
            predicted_indicies.append((j, prediction[0][j]))
    for j in predicted_indicies:
        print(j, end=", ")
    print()

    real_indicies = []
    print("Real keys:")
    for j in range(len(real_vectors[i])):
        if (real_vectors[i][j] == 1):
            real_indicies.append(j)
    print(real_indicies)

    for j in predicted_indicies:
        if j[0] not in real_indicies:
            a_l[0] += 1
    
    for j in range(len(predicted_indicies)):
        predicted_indicies[j] = predicted_indicies[j][0]

    for j in real_indicies:
        if j not in predicted_indicies:
            a_l[1] += 1
    i+=1

    added_lost_vector.append(a_l)

added = 0
lost = 0

for i in range(len(added_lost_vector)):
    added += added_lost_vector[i][0]
    lost += added_lost_vector[i][1]

    print(f"For file: {test_files[i]}")
    print(f"Predicted {added_lost_vector[i][0]} keys that were not pressed, and missed {added_lost_vector[i][1]} keys that were pressed.")

print(f"For all {number_of_files} files, the model added {added} keys that were not pressed, and missed {lost} keys that were pressed.")