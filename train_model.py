import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import os

# ==== Dataset Paths ====
train_dir = 'dataset/train'
test_dir = 'dataset/test'
model_save_path = 'pcb_defect_effnet_best.keras'

# ==== Model Settings ====
img_size = (300, 300)
batch_size = 16
epochs = 50

# ==== Data Augmentation ====
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    zoom_range=0.4,
    width_shift_range=0.25,
    height_shift_range=0.25,
    shear_range=0.2,
    brightness_range=[0.6, 1.4],
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

# ==== Load Data ====
train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

print(f"\n✅ Found {train_gen.samples} training images across {train_gen.num_classes} classes")
print(f"✅ Found {val_gen.samples} validation images")
print(f"✅ Defect classes: {list(train_gen.class_indices.keys())}\n")

# ==== Load EfficientNet Base ====
print("🔹 Loading EfficientNetB0 (pretrained on ImageNet)...")
base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=img_size + (3,))
base_model.trainable = False  # freeze base layers first

# Optionally unfreeze last few layers for fine-tuning
for layer in base_model.layers[-60:]:
    layer.trainable = True

# ==== Build Full Model ====
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
    layers.Dropout(0.3),
    layers.Dense(train_gen.num_classes, activation='softmax')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==== Training Callbacks ====
callbacks = [
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-7, verbose=1),
    ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True, verbose=1)
]

# ==== Train the Model ====
print("\n🚀 Starting model training...\n")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs,
    callbacks=callbacks,
    verbose=1
)

# ==== Evaluate on Test Set ====
val_loss, val_acc = model.evaluate(val_gen)
print(f"\n✅ Final Validation Accuracy: {val_acc * 100:.2f}%")
print(f"💾 Best model saved to: {model_save_path}")

# ==== Plot Training Curves ====
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend(); plt.title("Accuracy")

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend(); plt.title("Loss")

plt.show()
