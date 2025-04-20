# Suppress TensorFlow logging for cleaner output
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # '2' = show only warnings and errors

#bech t excuti l fichier f terminal tkteb lcommande hevi 
# awel wahda heya l interpreter chemin d acces w thenya taa file l fih lcode heva 
# C:/Users/habib/new_cnn_venv/Scripts/python.exe "c:/ISG/2BIS/probleme solving/DS2/cnn_train.py"

# Import necessary libraries
import tensorflow as tf
from tensorflow.keras import layers, models  # For building CNN architecture
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # For loading and augmenting image data
import matplotlib.pyplot as plt  # For plotting training results

# === 1. Configuration ===

# Set path to image dataset (must be organized in subfolders per class)
DATA_DIR = r"C:\ISG\2BIS\probleme solving\DS2\data\images"

# Define image size (height, width) and batch size
IMG_SIZE = (150, 150)       # All images will be resized to this size
BATCH_SIZE = 32             # Number of images per batch during training
EPOCHS = 10                 # Number of training cycles over the dataset

# === 2. Data Preparation ===

# Create an ImageDataGenerator for training (includes data augmentation)
train_datagen = ImageDataGenerator(
    rescale=1./255,         # Normalize pixel values between 0 and 1
    validation_split=0.2,   # Reserve 20% of data for validation
    rotation_range=20,      # Randomly rotate images (data augmentation)
    horizontal_flip=True    # Randomly flip images horizontally
)

# Load training data from directory
train_generator = train_datagen.flow_from_directory(
    DATA_DIR,                        # Path to dataset
    target_size=IMG_SIZE,           # Resize images
    batch_size=BATCH_SIZE,          # Batch size
    class_mode='categorical',       # For multi-class classification
    subset='training',              # Use 80% of data for training
    shuffle=True                    # Shuffle images
)

# Load validation data from same directory
val_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',            # Use 20% of data for validation
    shuffle=False                   # Don't shuffle validation data
)

# === 3. CNN Model Architecture ===

# Build a simple Convolutional Neural Network (CNN)
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(*IMG_SIZE, 3)),  # 1st conv layer with 32 filters
    layers.MaxPooling2D((2,2)),                                               # 1st pooling layer
    layers.Conv2D(64, (3,3), activation='relu'),                              # 2nd conv layer with 64 filters
    layers.MaxPooling2D((2,2)),                                               # 2nd pooling layer
    layers.Flatten(),                                                         # Flatten the 2D output to 1D
    layers.Dense(128, activation='relu'),                                     # Fully connected layer with 128 neurons
    layers.Dense(train_generator.num_classes, activation='softmax')          # Output layer (softmax for multi-class)
])

# === 4. Model Compilation ===

# Compile the model with optimizer, loss function, and evaluation metric
model.compile(
    optimizer='adam',                        # Adaptive optimizer
    loss='categorical_crossentropy',         # Suitable for multi-class classification
    metrics=['accuracy']                     # Metric to track
)

# === 5. Model Training ===

# Train the model on training data and validate on validation data
history = model.fit(
    train_generator,             # Training data
    validation_data=val_generator,  # Validation data
    epochs=EPOCHS                # Number of epochs
)

# === 6. Model Saving ===

# Save the trained model to a file (can use .h5 or .keras extension)
model.save('food_classifier.keras')

# === 7. Visualization of Training Performance ===

# Create a new figure for plotting with size and resolution
plt.figure(figsize=(10, 6), dpi=100)

# Plot training accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue', marker='o')

# Plot validation accuracy
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='green', marker='o')

# Plot training loss
plt.plot(history.history['loss'], label='Training Loss', color='red', linestyle='--', marker='x')

# Plot validation loss
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange', linestyle='--', marker='x')

# Add chart title and axis labels
plt.title("Model Performance Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy / Loss")

# Add grid and legend for better readability
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Adjust layout to prevent text cutoff
plt.tight_layout()

# Save the plot to a high-resolution image file
plt.savefig("training_results.png", dpi=300)

# Show the plot on screen
plt.show()

# Print success message
print("✅ Modèle sauvegardé sous 'food_classifier.keras' et graphique sous 'training_results.png' !")
