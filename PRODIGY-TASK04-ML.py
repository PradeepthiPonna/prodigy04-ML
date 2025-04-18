import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Paths to your CSV files
train_csv_path = r"C:\Users\PRADEEPTHI\Downloads\archive (7)\sign_mnist_train.csv"
test_csv_path = r"C:\Users\PRADEEPTHI\Downloads\archive (7)\sign_mnist_test.csv"

# Load the data
train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)

print(f"Training data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")

# Prepare data
def prepare_data(df):
    # Extract labels and convert to one-hot encoding
    y = to_categorical(df['label'], num_classes=25)
    
    # Drop label column and reshape images (28x28 pixels)
    X = df.drop('label', axis=1).values.reshape(-1, 28, 28, 1)
    
    # Normalize pixel values
    X = X / 255.0
    
    return X, y

X_train, y_train = prepare_data(train_df)
X_test, y_test = prepare_data(test_df)

# Split training data for validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

print(f"\nFinal shapes:")
print(f"Train: {X_train.shape}, {y_train.shape}")
print(f"Validation: {X_val.shape}, {y_val.shape}")
print(f"Test: {X_test.shape}, {y_test.shape}")

# Model architecture
def create_model():
    model = Sequential([
        Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        
        Conv2D(128, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        
        Conv2D(256, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(25, activation='softmax')  # 25 classes (a-y, no j or z)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

model = create_model()
model.summary()

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

# Train the model
history = model.fit(X_train, y_train,
                    batch_size=64,
                    epochs=30,
                    validation_data=(X_val, y_val),
                    callbacks=[early_stop, reduce_lr])

# Evaluation
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc*100:.2f}%")

# Plot training history
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss')
plt.legend()
plt.show()

# Save the model
model.save('asl_sign_model.h5')
print("Model saved as 'asl_sign_model.h5'")

# Class mapping (ASL letters a-y, excluding j and z)
asl_classes = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 
    'T', 'U', 'V', 'W', 'X', 'Y'
]

# Prediction function
def predict_asl_sign(image_array, model, classes):
    """Predict ASL sign from preprocessed image array"""
    if len(image_array.shape) == 3:
        image_array = np.expand_dims(image_array, axis=0)
    
    preds = model.predict(image_array)
    pred_class = classes[np.argmax(preds)]
    confidence = np.max(preds)
    
    return pred_class, confidence

# Example usage (with a test image)
sample_idx = 42  # Try different indices
sample_image = X_test[sample_idx]
true_label = asl_classes[np.argmax(y_test[sample_idx])]

pred_letter, confidence = predict_asl_sign(sample_image, model, asl_classes)
print(f"\nSample Prediction:")
print(f"True Label: {true_label}")
print(f"Predicted: {pred_letter} ({confidence*100:.1f}% confidence)")

# Display sample image
plt.imshow(sample_image.squeeze(), cmap='gray')
plt.title(f"True: {true_label} | Pred: {pred_letter}")
plt.axis('off')
plt.show()