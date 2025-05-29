import sounddevice as sd
import numpy as np
import pygame
import tensorflow.keras as keras
from model import focal_loss_custom
from generate_spec import generate_mel_spectrogram

model = keras.models.load_model(
    "models/piano_note_model_2048_v0.8.keras",
    custom_objects={'custom_loss': focal_loss_custom}
)

SAMPLE_RATE = 44100
SEGMENT_DURATION = 0.05
SAMPLES_PER_SEGMENT = int(SAMPLE_RATE * SEGMENT_DURATION)

pygame.init()
WHITE_KEY_WIDTH = 22.5  
BLACK_KEY_WIDTH = 12
WHITE_KEY_HEIGHT = 85
BLACK_KEY_HEIGHT = 50

TOTAL_WHITE_KEYS = 52
WIDTH = int(TOTAL_WHITE_KEYS * WHITE_KEY_WIDTH)  
HEIGHT = WHITE_KEY_HEIGHT

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("")
clock = pygame.time.Clock()

MIDI_START = 21  
MIDI_END = 108   

BLACK_NOTES = {1, 3, 6, 8, 10}  

key_positions = []
white_index = 0

for midi_note in range(MIDI_START, MIDI_END + 1):
    note_in_octave = midi_note % 12
    is_black = note_in_octave in BLACK_NOTES
    
    if not is_black:
        x_pos = round(white_index * WHITE_KEY_WIDTH)  
        key_positions.append((midi_note - MIDI_START, x_pos, is_black))
        white_index += 1
    else:
        x_pos = round(white_index * WHITE_KEY_WIDTH - BLACK_KEY_WIDTH // 2)
        key_positions.append((midi_note - MIDI_START, x_pos, is_black))

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
ACTIVE_WHITE = (255, 100, 100)  
ACTIVE_BLACK = (200, 0, 0)      
BACKGROUND = (240, 240, 240)    

def draw_keyboard(active_keys):
    screen.fill(BACKGROUND)
    
    pygame.draw.rect(screen, WHITE, (0, 0, WIDTH, WHITE_KEY_HEIGHT))
    
    for i in range(1, TOTAL_WHITE_KEYS):
        x_pos = round(i * WHITE_KEY_WIDTH)
        pygame.draw.line(screen, BLACK, (x_pos, 0), (x_pos, WHITE_KEY_HEIGHT), 1)
    
    for note, x_pos, is_black in key_positions:
        if not is_black and note in active_keys:
            pygame.draw.rect(screen, ACTIVE_WHITE, (x_pos, 0, WHITE_KEY_WIDTH, WHITE_KEY_HEIGHT))
    
    for note, x_pos, is_black in key_positions:
        if is_black:
            color = ACTIVE_BLACK if note in active_keys else BLACK
            pygame.draw.rect(screen, color, (x_pos, 0, BLACK_KEY_WIDTH, BLACK_KEY_HEIGHT))
    
    pygame.display.flip()

try:
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt

        audio = sd.rec(SAMPLES_PER_SEGMENT, samplerate=SAMPLE_RATE, channels=1, dtype='int16')
        sd.wait()
        audio_data = audio.flatten().astype(np.float32) / 32768.0
        spec = generate_mel_spectrogram(audio_data, sr=SAMPLE_RATE)
        spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)
        spec = np.expand_dims(spec, axis=(0, -1))

        predictions = model.predict(spec, verbose=0)[0]
        
        max_pred = predictions.max()
        threshold = max_pred * 0.5 if max_pred > 0.2 else 0.1
        
        active_keys = [i for i, pred in enumerate(predictions) if pred > threshold]
        
        draw_keyboard(active_keys)

        clock.tick(20)

except KeyboardInterrupt:
    pygame.quit()