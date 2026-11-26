import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models

def get_model():
    # Load EfficientNetB0 pretrained on ImageNet
    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(256, 256, 3)
    )

    # Initially freeze the base model
    base_model.trainable = False

    inputs = layers.Input(shape=(256, 256, 3))

    x = base_model(inputs, training=False)  # IMPORTANT for batchnorm stability
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs)

    # Compile for initial training (Phase 1)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model
