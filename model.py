import numpy as np
import os
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras import layers, regularizers, Model, Input
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.losses import SigmoidFocalCrossEntropy
from tensorflow_addons.layers import GroupNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger

print("Dostępne urządzenia:", tf.config.list_physical_devices())
print(tf.__version__)

def se_block(input_tensor, reduction=16):
    filters = input_tensor.shape[-1]
    se = layers.GlobalAveragePooling2D()(input_tensor)
    se = layers.Dense(filters // reduction, activation='relu')(se)
    se = layers.Dense(filters, activation='sigmoid')(se)
    se = layers.Reshape((1, 1, filters))(se)
    return layers.multiply([input_tensor, se])

def conv_block(x, filters):
    shortcut = x
    x = layers.Conv2D(filters, (5,1 ), padding='same', use_bias=False)(x)
    x = GroupNormalization(groups=8)(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, (3,1 ), padding='same', use_bias=False)(x)
    x = GroupNormalization(groups=8)(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, (1,1 ), padding='same', use_bias=False)(x) #
    x = GroupNormalization(groups=8)(x) #
    x = layers.Activation('relu')(x) #
    x = se_block(x)
    if x.shape[-1] != shortcut.shape[-1]:  
        shortcut = layers.Conv2D(filters, (1,1), padding='same')(shortcut)
    x = layers.add([x, shortcut])
    # if x.shape[2] > 1:
    #     x = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(x)
    # else:
    #     x = layers.MaxPooling2D(pool_size=(2, 1), padding="same")(x)
    if x.shape[2] > 2:
        x = layers.AveragePooling2D(pool_size=(1, 2), padding="same")(x)
    return x

def focal_loss_custom(gamma=1.25, alpha=0.25, neighbor_weight=0.2, fn_weight=0.9):
    focal_loss_fn = SigmoidFocalCrossEntropy(gamma=gamma, alpha=alpha)

    def custom_loss(y_true, y_pred):
        focal = focal_loss_fn(y_true, y_pred)

        penalty = 0.0
        for shift in [-1, 1]:
            shifted_true = tf.roll(y_true, shift=shift, axis=1)
            shifted_true = tf.where(
                tf.logical_or(shifted_true > 1, shifted_true < 0),
                tf.zeros_like(shifted_true),
                shifted_true
            )

            mask = tf.logical_and(shifted_true > 0, y_true == 0)
            penalty += tf.reduce_sum(tf.where(mask, y_pred, 0.0), axis=1)

        fn_mask = tf.logical_and(y_true == 1, y_pred < 0.2)
        fn_penalty = tf.reduce_sum(tf.where(fn_mask, 1.0 - y_pred, 0.0), axis=1)

        final_loss = focal + neighbor_weight * penalty + fn_weight * fn_penalty
        return final_loss

    return custom_loss

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

if __name__ == "__main__":
    X_train = np.load("tmp/X_train.npy", mmap_mode='r')
    y_train = np.load("tmp/y_train.npy", mmap_mode='r')
    X_test = np.load("tmp/X_test.npy", mmap_mode='r')
    y_test = np.load("tmp/y_test.npy", mmap_mode='r')

    x_train_shape = np.load("tmp/X_train.npy", mmap_mode='r').shape[1:]
    y_train_shape = np.load("tmp/y_train.npy", mmap_mode='r').shape[1:]

    input_shape = X_train.shape[1:]
    inputs = Input(shape=input_shape)

    x = layers.Conv2D(32, (3,1 ), padding='same', use_bias=False)(inputs)
    x = GroupNormalization(groups=8)(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=(1, 2), padding="same")(x)

    for filters in [32, 64, 128, 256, 512, 512]:
        x = conv_block(x, filters)

    x = layers.Flatten()(x)
    # x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(88, activation='sigmoid', 
                        kernel_regularizer=regularizers.l1(1e-5))(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=focal_loss_custom(gamma=1.25, alpha=0.25),
        metrics=[
            'accuracy',
            AUC(name='auc', num_thresholds=200, multi_label=True),
            Precision(name='precision', thresholds=[0.2]),
            Recall(name='recall', thresholds=[0.2])
        ],
    )

    csv_logger = CSVLogger("training_log.csv", append=True)

    callbacks = [
        ReduceLROnPlateau(monitor="val_auc", factor=0.5, patience=3, verbose=1),
       # EarlyStopping(monitor="val_auc", patience=15, min_delta=0.0005, restore_best_weights=True, verbose=1),
        ModelCheckpoint("models/piano_note_model_2048_v0.8.keras", 
                        monitor='val_auc',
                        save_best_only=True,
                        mode='max',
                        verbose=1),
        csv_logger,
    ]

    model.summary()

    batch_size = 32

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
        epochs=50,
        callbacks=callbacks,
        verbose=1
    )

    test_results = model.evaluate(test_dataset, steps=validation_steps)

    print(f"Test Loss: {test_results[0]:.4f}")
    print(f"Test Accuracy: {test_results[1]:.4f}")
    print(f"Test AUC: {test_results[2]:.4f}")