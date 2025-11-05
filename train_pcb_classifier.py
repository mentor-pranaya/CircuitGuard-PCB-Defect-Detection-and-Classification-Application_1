import tensorflow as tf
from tensorflow.keras import layers, models

# Path to your images folder
data_path = "dataset/test_images"  # update with your actual path if needed
img_height, img_width = 128, 128

# Data augmentation block
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
])

# Load training and validation data
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=32
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=32
)
num_classes = len(train_ds.class_names)

# Base MobileNetV2 model
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(img_height, img_width, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

# Build the model
model = models.Sequential([
    data_augmentation,
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Training with frozen base model...")
model.fit(train_ds, validation_data=val_ds, epochs=10)

# Fine-tune: unfreeze the base model and train further
base_model.trainable = True
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # Lower learning rate
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Fine-tuning with unfrozen base model...")
model.fit(train_ds, validation_data=val_ds, epochs=10)

# Save in both formats
model.save('pcb_defect_classifier.keras')
model.save('pcb_defect_classifier.h5')
