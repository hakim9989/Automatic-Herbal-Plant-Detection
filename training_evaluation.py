import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Resizing, Rescaling, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, log_loss
import numpy as np
import os
import seaborn as sns
import pandas as pd

# --- Configuration ---
# IMPORTANT: Ensure these values match the actual characteristics of your dataset
img_height, img_width = 224, 224 # Standard input size for ResNet50
num_classes = 6 # <<< IMPORTANT: Set this to your actual number of plant categories
class_names = ['Betel', 'Guava', 'Lemon', 'Mint', 'Neem', 'Tulsi'] # <<< IMPORTANT: List your actual class names in order

# Define dataset path where your raw data (folders per class) is located
dataset_base_path = r'D:\project\auto'

# Output directories
# IMPORTANT: Define a base directory for all outputs.
# This should be a path where you have write permissions.
BASE_DIR = 'D:/project/resnet_output' 
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

# Create directories if they don't exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# --- 1. Data Loading and Splitting ---
print("--- Loading Dataset ---")
dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_base_path,
    shuffle=True,
    batch_size=32, # Use the batch_size defined in config
    image_size=(img_height, img_width),
    label_mode='int' # Ensure labels are integers for SparseCategoricalCrossentropy
)

# Class Information (from the loaded dataset)
# Ensure class_names and num_classes are correctly inferred from the dataset
# if they were not hardcoded above.
class_names = dataset.class_names
num_classes = len(class_names)
print("Class Names:", class_names)
print("Number of Classes:", num_classes)

# Find the batch size and image size (for verification)
for images, labels in dataset.take(1):
    print("Batch Size (actual):", images.shape[0])
    print("Image Size (actual):", images.shape[1:])
    print("Image Data Type (raw):", images.dtype)
    print("Label Data Type (raw):", labels.dtype)
    print("Label Shape (raw):", labels.shape)
    print("Labels in Batch (raw):", labels.numpy())

# Data Split Function
def get_dataset_partisions_tf(ds, train_split=0.75, val_split=0.15, test_split=0.1, shuffle=True, shuffle_size=10000):
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=42) # Using a fixed seed for reproducibility
    dataset_size = len(ds)
    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)

    train_ds = ds.take(train_size)
    remaining_ds = ds.skip(train_size)
    val_ds = remaining_ds.take(val_size)
    test_ds = remaining_ds.skip(val_size) # The rest of the dataset is for testing

    return train_ds, val_ds, test_ds

train_data, val_data, test_data = get_dataset_partisions_tf(dataset)
print(f"Train Data Batches: {len(train_data)}, Validation Data Batches: {len(val_data)}, Test Data Batches: {len(test_data)}")

# --- 2. Preprocessing Layer (Applied within the dataset pipeline) ---
# ResNet models typically expect input pixels to be normalized to a specific range.
# We'll rescale to [0, 1] for general compatibility.
resize_and_rescale = Sequential([
    Resizing(img_height, img_width),
    Rescaling(1./255) # Normalize to [0, 1]
])

# Apply preprocessing to datasets
# This maps the preprocessing function over each batch in the dataset
AUTOTUNE = tf.data.AUTOTUNE # For performance optimization
train_data = train_data.map(lambda x, y: (resize_and_rescale(x), y), num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)
val_data = val_data.map(lambda x, y: (resize_and_rescale(x), y), num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)
test_data = test_data.map(lambda x, y: (resize_and_rescale(x), y), num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)


# --- 3. Build ResNet50 Model ---
print("\n--- Building ResNet50 Model ---")
base_model = ResNet50(
    weights='imagenet',
    input_shape=(img_height, img_width, 3), # Ensure this matches your preprocessed image size
    include_top=False, # Exclude the top (classification) layers
    pooling='avg' # Use Global Average Pooling
)

# --- 4. Constructing the Model (with Custom Head) ---
base_model.trainable = False # Freeze the base model initially

inputs = tf.keras.Input(shape=(img_height, img_width, 3)) # Input shape for the overall model
# The preprocessing (resizing/rescaling) is handled by the tf.data.Dataset pipeline
x = base_model(inputs, training=False) # Pass through the frozen base model
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
outputs = Dense(num_classes, activation='softmax')(x) # num_classes from data setup
model = Model(inputs, outputs)

# --- 5. Compile the Model ---
model.compile(
    optimizer=Adam(), # Using Adam optimizer
    loss=SparseCategoricalCrossentropy(), # For integer labels (label_mode='int' from image_dataset_from_directory)
    metrics=[SparseCategoricalAccuracy()] # For integer labels
)

# --- 6. Layer Description of the Model ---
print("\n--- Model Summary ---")
model.summary()

# --- 7. Set up Callbacks ---
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=1e-6,
    verbose=1
)
model_checkpoint = ModelCheckpoint(
    filepath=os.path.join(CHECKPOINT_DIR, 'best_resnet_model.keras'),
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

callbacks = [early_stopping, reduce_lr, model_checkpoint]

# --- 8. Training the Model ---
print("\n--- Training Model ---")
epochs = 30 # Number of epochs for training
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs,
    callbacks=callbacks
)
print("\nModel training finished.")

# --- 9. Evaluating the Test Data ---
print("\n--- Evaluating Test Data ---")
loss, accuracy = model.evaluate(test_data)
print(f"Test Loss (Keras evaluate): {loss:.4f}")
print(f"Test Accuracy (Keras evaluate): {accuracy:.4f}")

# --- 10. Saving the Model ---
model_save_path = os.path.join(BASE_DIR, "Trained_ResNet_model.keras")
model.save(model_save_path)
print(f"Model saved to {model_save_path}\n")

# --- 11. Generate Predictions and Calculate Detailed Metrics ---
print("--- Generating Predictions for Detailed Metrics ---")
# Collect true labels and predictions from the test_data
y_true_list = []
y_pred_prob_list = []

# Iterate through the test_data batches
for images, labels in test_data:
    y_true_list.extend(labels.numpy())
    y_pred_prob_list.extend(model.predict(images))

y_true = np.array(y_true_list)
y_pred_prob = np.array(y_pred_prob_list)
y_pred = np.argmax(y_pred_prob, axis=1) # Convert probabilities to class labels

# Calculate detailed metrics using scikit-learn
accuracy_sk = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
# log_loss needs one-hot true labels
y_true_one_hot_for_logloss = tf.keras.utils.to_categorical(y_true, num_classes=num_classes)
logloss_sk = log_loss(y_true_one_hot_for_logloss, y_pred_prob)

print("\n--- Detailed Evaluation Metrics (Scikit-learn) ---")
print(f"Accuracy: {accuracy_sk:.4f}")
print(f"Precision (weighted): {precision:.4f}")
print(f"Recall (weighted): {recall:.4f}")
print(f"F1-Score (weighted): {f1:.4f}")
print(f"Log Loss: {logloss_sk:.4f}")

# Classification Report
classification_rep = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
print("\n--- Classification Report ---")
print(classification_rep)

# Confusion Matrix Plot
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (Test Set)')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix_resnet.png'))
plt.show()

# --- 12. Plotting Accuracy and Loss Graphs ---
print("\n--- Plotting Training History ---")
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'training_plots_resnet.png'))
plt.show()

# --- 13. Save Training History ---
history_df = pd.DataFrame(history.history)
history_df.to_csv(os.path.join(RESULTS_DIR, 'training_history_resnet.csv'), index=True)
print(f"Training history saved to '{os.path.join(RESULTS_DIR, 'training_history_resnet.csv')}'")

print("\nResNet50 model training and evaluation complete.")
