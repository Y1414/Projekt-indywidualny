import numpy as np
import os
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras import layers, regularizers, Model, Input
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.losses import SigmoidFocalCrossEntropy
from tensorflow_addons.layers import GroupNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

print("Dostępne urządzenia:", tf.config.list_physical_devices())
print(tf.__version__)

def se_block(input_tensor, reduction=16):
    filters = input_tensor.shape[-1]
    se = layers.GlobalAveragePooling2D()(input_tensor)
    se = layers.Dense(filters // reduction, activation='relu')(se)
    se = layers.Dense(filters, activation='sigmoid')(se)
    se = layers.Reshape((1,1,filters))(se)
    return layers.multiply([input_tensor, se])

def conv_block(x, filters):
    shortcut = x
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = GroupNormalization(groups=8)(x)
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = GroupNormalization(groups=8)(x)
    x = se_block(x)
    if x.shape == shortcut.shape:
        x = layers.add([x, shortcut])
    if x.shape[2] > 1:
        x = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(x)
    else:
        x = layers.MaxPooling2D(pool_size=(2, 1), padding="same")(x)
    return x

X_train = np.load("tmp/X_train.npy", mmap_mode='r')
y_train = np.load("tmp/y_train.npy", mmap_mode='r')
X_test = np.load("tmp/X_test.npy", mmap_mode='r')
y_test = np.load("tmp/y_test.npy", mmap_mode='r')

x_train_shape = np.load("tmp/X_train.npy", mmap_mode='r').shape[1:]
y_train_shape = np.load("tmp/y_train.npy", mmap_mode='r').shape[1:]

input_shape = X_train.shape[1:]
inputs = Input(shape=input_shape)

x = layers.Conv2D(32, 7, padding='same', activation='relu')(inputs)
x = GroupNormalization(groups=8)(x)
x = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(x)

for filters in [32, 64, 128, 128]:
    x = conv_block(x, filters)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-3))(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(88, activation='sigmoid', 
                      kernel_regularizer=regularizers.l1(1e-5))(x)

model = Model(inputs, outputs)

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss=SigmoidFocalCrossEntropy(reduction="auto", gamma=1.25, alpha=0.25),
    metrics=[
        'accuracy',
        AUC(name='auc', num_thresholds=200, multi_label=True),
        Precision(name='precision', thresholds=[0.2]),
        Recall(name='recall', thresholds=[0.2])
    ],
)

callbacks = [
    ReduceLROnPlateau(monitor="val_auc", factor=0.5, patience=3, verbose=1),
    EarlyStopping(monitor="val_auc", patience=18, restore_best_weights=True, verbose=1),
    ModelCheckpoint("models/piano_note_model_commercial_v0.6.keras", 
                    monitor='val_auc',
                    save_best_only=True,
                    mode='max',
                    verbose=1)
]

model.summary()

batch_size = 32

def data_generator(x_path, y_path, batch_size=32, shuffle=True):
    x_data = np.load(x_path, mmap_mode='r')
    y_data = np.load(y_path, mmap_mode='r')
    num_samples = len(x_data)
    indices = np.arange(num_samples)
    if shuffle:
        np.random.shuffle(indices)
    while True:
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_x = x_data[batch_indices]
            batch_y = y_data[batch_indices]
            yield batch_x, batch_y

train_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator("tmp/X_train.npy", "tmp/y_train.npy", batch_size=batch_size, shuffle=True),
    output_signature=(
        tf.TensorSpec(shape=(None,) + input_shape, dtype=tf.float32),
        tf.TensorSpec(shape=(None, 88), dtype=tf.float32)
    )
).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator("tmp/X_test.npy", "tmp/y_test.npy", batch_size=batch_size, shuffle=False),
    output_signature=(
        tf.TensorSpec(shape=(None,) + input_shape, dtype=tf.float32),
        tf.TensorSpec(shape=(None, 88), dtype=tf.float32)
    )
).prefetch(tf.data.AUTOTUNE)

steps_per_epoch = len(X_train) // batch_size
validation_steps = int(np.ceil(len(X_test) / batch_size))

history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    epochs=30,
    callbacks=callbacks,
    verbose=1
)

test_results = model.evaluate(test_dataset, steps=validation_steps)

print(f"Test Loss: {test_results[0]:.4f}")
print(f"Test Accuracy: {test_results[1]:.4f}")
print(f"Test AUC: {test_results[2]:.4f}")