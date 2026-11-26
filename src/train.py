import os
import tensorflow as tf
from preprocess import load_image
from model import get_model

# ❗ UPDATE THESE PATHS for the full MICC-F2000 dataset later
ORIGINAL_PATH = "/Users/venkatsomanadh/Downloads/Forgery Detection/dataset/original_full"
FORGED_PATH   = "/Users/venkatsomanadh/Downloads/Forgery Detection/dataset/forged_full"

print("Original path exists:", os.path.exists(ORIGINAL_PATH))
print("Forged path exists:", os.path.exists(FORGED_PATH))

def load_dataset(original_dir, forged_dir):
    orig_paths = [os.path.join(original_dir, f) for f in os.listdir(original_dir)]
    forg_paths = [os.path.join(forged_dir, f) for f in os.listdir(forged_dir)]

    labels = [0]*len(orig_paths) + [1]*len(forg_paths)
    all_paths = orig_paths + forg_paths

    ds = tf.data.Dataset.from_tensor_slices((all_paths, labels))

    # Shuffle harder for 2000 images
    ds = ds.shuffle(buffer_size=5000)

    # 🔥 Strong augmentation (best for large dataset)
    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomContrast(0.2),
        tf.keras.layers.RandomBrightness(0.2),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
    ])

    ds = ds.map(lambda x, y: (augment(load_image(x)), y),
                num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(32).prefetch(tf.data.AUTOTUNE)

    return ds


if __name__ == "__main__":

    train_ds = load_dataset(ORIGINAL_PATH, FORGED_PATH)

    model = get_model()

    # 🔥 Fine-tune EfficientNet deeper after initial training
    model.fit(train_ds, epochs=10)   # Phase 1 (just classifier training)

    # ➤ UNFREEZE LAST 50% OF EfficientNet for fine-tuning
    base_model = model.layers[0]
    base_model.trainable = True

    # Use slow learning rate for safe fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # 🔥 Phase 2 — Fine tune entire model
    model.fit(train_ds, epochs=20)

    model.save("/Users/venkatsomanadh/Downloads/Forgery Detection/forgery_classifier_final.h5")
    print("🔥 FINAL MODEL TRAINING COMPLETE!")
