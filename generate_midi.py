import random
import numpy as np
import pretty_midi
from multiprocessing import Pool, cpu_count

def generate_sample(file_index):
    total_duration = 2500
    bpm = 120
    note_duration_quarter = 0.25
    seconds_per_beat = 60 / bpm
    note_duration_sec = note_duration_quarter * seconds_per_beat
    num_notes = int(total_duration * 2 / note_duration_quarter)
    print(num_notes)
    exit(0)
    midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    instrument = pretty_midi.Instrument(program=0)
    labels = []

    time = 0.0

    for i in range(num_notes):
        num_pitches =  random.choice(range(0, 8))
        pitches = random.sample(range(21, 109), num_pitches)
        keys_pressed = [0] * 88

        for pitch in pitches:
            note = pretty_midi.Note(velocity=100, pitch=pitch,
                                    start=time, end=time + note_duration_sec)
            instrument.notes.append(note)
            keys_pressed[pitch - 21] = 1

        labels.append(keys_pressed)
        time += note_duration_sec

        if i % 1000 == 0:
            print(f'File {file_index}: {i}/{num_notes}')

    midi.instruments.append(instrument)
    midi.write(f'{file_index}.mid')
    np.save(f'{file_index}.npy', np.array(labels))

    print(f"File {file_index} saved.")
    return file_index


if __name__ == '__main__':
    num_files = 5
    with Pool(processes=min(cpu_count(), num_files)) as pool:
        pool.map(generate_sample, range(num_files))
