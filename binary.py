import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import os

# === Configuration ===
# Path to your organized dataset
data_dir = "C:/project/herbal"
# Image dimensions for the model input
img_height = 224
img_width = 224
batch_size = 32
epochs = 27 # You might need more epochs depending on your dataset size and complexity
validation_split = 0.2 # 20% of the data will be used for validation
random_seed = 42 # For reproducibility
model_save_path = "is_plant_model.keras" # The model file to save

# === Data Loading and Preprocessing ===
# Using ImageDataGenerator for efficient loading and augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255, # Normalize pixel values to [0, 1]
    validation_split=validation_split, # Split for validation data
    # Optional: Add data augmentation for better generalization
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Create training generator
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary', # IMPORTANT: 'binary' for two classes
    subset='training', # Specify this as the training subset
    seed=random_seed
)

# Create validation generator
validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation', # Specify this as the validation subset
    seed=random_seed
)

# Check class indices (should be {'non_plant': 0, 'plant': 1} or {'plant': 0, 'non_plant': 1})
print("Class Indices:", train_generator.class_indices)
# Make sure 'plant' corresponds to the class you want to predict as 1 (or close to 1)

# === Model Architecture (Simple CNN for Binary Classification) ===
def build_binary_classifier_model(img_height, img_width):
    """
    Builds a simple CNN model for binary classification (plant vs. non-plant).
    Outputs a single probability using sigmoid activation.
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5), # Dropout for regularization to prevent overfitting
        layers.Dense(1, activation='sigmoid') # Single output for binary classification (probability)
    ])
    return model

# Build the model
model = build_binary_classifier_model(img_height, img_width)

# Compile the model
model.compile(
    optimizer='adam', # Adam optimizer is a good default
    loss='binary_crossentropy', # Appropriate loss for binary classification
    metrics=['accuracy'] # Monitor accuracy during training
)

model.summary()

# === Model Training ===
print("\nStarting model training...")
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True), # Stop if val_loss doesn't improve for 10 epochs
        tf.keras.callbacks.ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_loss') # Save the best model based on validation loss
    ]
)
print("Model training finished.")

# === Model Evaluation (Optional, on validation data) ===
print("\nEvaluating final model on validation data...")
loss, accuracy = model.evaluate(validation_generator)
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")

print(f"\nBinary classification model saved to {model_save_path}")
