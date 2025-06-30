import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from scipy.signal import spectrogram
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


# Parameters
train_dir = "/home/nader/gnss_dataset/Raw_IQ_Dataset/Training"
test_dir = "/home/nader/gnss_dataset/Raw_IQ_Dataset/Testing"
classes = ['DME', 'NB', 'NoJam', 'SingleAM', 'SingleChirp', 'SingleFM']
img_size = (128, 128)

def load_and_preprocess_data(data_dir, classes, img_size=(128, 128)):
    X, y = [], []
    for cls in classes:
        folder = os.path.join(data_dir, cls)
        for file in os.listdir(folder):
            path = os.path.join(folder, file)
            try:
                mat = sio.loadmat(path)
                if 'GNSS_plus_Jammer_awgn' not in mat:
                    continue
                iq = mat['GNSS_plus_Jammer_awgn'].squeeze()
                f, t, Sxx = spectrogram(iq, fs=1e6, nperseg=256, noverlap=128)
                Sxx_log = np.log10(Sxx + 1e-10)
                Sxx_resized = cv2.resize(Sxx_log, img_size)
                X.append(Sxx_resized)
                y.append(cls)
            except Exception as e:
                print(f"Error processing {path}: {e}")
    return np.array(X), np.array(y)

# Load datasets
X_train_raw, y_train = load_and_preprocess_data(train_dir, classes, img_size)
X_test_raw, y_test = load_and_preprocess_data(test_dir, classes, img_size)

# Normalization
X_min, X_max = X_train_raw.min(), X_train_raw.max()
X_train = (X_train_raw - X_min) / (X_max - X_min)
X_test = (X_test_raw - X_min) / (X_max - X_min)
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# Data Augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)
datagen.fit(X_train)

# Labels
le = LabelEncoder()
y_train_enc = to_categorical(le.fit_transform(y_train))
y_test_enc = to_categorical(le.transform(y_test))

# CNN Model (based on the GitHub repository)
'''
def create_cnn_model(input_shape=(128,128,1), num_classes=6):
    m = models.Sequential()
    m.add(layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape))
    m.add(layers.BatchNormalization())
    m.add(layers.MaxPooling2D((2,2)))

    m.add(layers.Conv2D(64, (3,3), activation='relu', padding='same'))
    m.add(layers.BatchNormalization())
    m.add(layers.MaxPooling2D((2,2)))

    m.add(layers.Conv2D(128, (3,3), activation='relu', padding='same'))
    m.add(layers.BatchNormalization())
    m.add(layers.MaxPooling2D((2,2)))

    m.add(layers.Flatten())
    m.add(layers.Dense(128, activation='relu'))
    m.add(layers.Dropout(0.5))
    m.add(layers.Dense(64, activation='relu'))  # ✅ extra layer
    m.add(layers.Dropout(0.3))                  # ✅ extra dropout
    m.add(layers.Dense(num_classes, activation='softmax'))
    return m
'''

def create_cnn_model(input_shape=(128,128,1), num_classes=6):
    m = models.Sequential()
    m.add(layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape))
    m.add(layers.BatchNormalization())
    m.add(layers.MaxPooling2D((2,2)))

    m.add(layers.Conv2D(64, (3,3), activation='relu', padding='same'))
    m.add(layers.BatchNormalization())
    m.add(layers.MaxPooling2D((2,2)))

    m.add(layers.Conv2D(128, (3,3), activation='relu', padding='same'))
    m.add(layers.BatchNormalization())
    m.add(layers.MaxPooling2D((2,2)))

    m.add(layers.Flatten())
    m.add(layers.Dense(128, activation='relu'))
    m.add(layers.Dropout(0.2))
    m.add(layers.Dense(64, activation='relu'))  # ✅ extra layer
    #m.add(layers.Dropout(0.2))                  # ✅ extra dropout
    m.add(layers.Dense(num_classes, activation='softmax'))
    return m


cnn_model = create_cnn_model()

# Compile
cnn_model.compile(optimizer=Nadam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-10, verbose=1)

# Train

history = cnn_model.fit(X_train, y_train_enc,
                        validation_data=(X_test, y_test_enc),
                        epochs=50,
                        batch_size=8,
                        callbacks=[early_stop, reduce_lr],
                        verbose=2)
'''

history = cnn_model.fit(
    datagen.flow(X_train, y_train_enc, batch_size=32),
    validation_data=(X_test, y_test_enc),
    epochs=50,
    callbacks=[early_stop, reduce_lr],
    verbose=2
)
'''
# Evaluate
test_loss, test_accuracy = cnn_model.evaluate(X_test, y_test_enc, verbose=0)
results_df = pd.DataFrame({
    "Test Accuracy": [test_accuracy],
    "Test Loss": [test_loss]
})

# Plotting
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
plt.tight_layout()
plt.show()


print(results_df)



y_pred = cnn_model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test_enc, axis=1)

print(classification_report(y_true_labels, y_pred_labels, target_names=classes))

conf_mat = confusion_matrix(y_true_labels, y_pred_labels)
plt.figure(figsize=(8,6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix")
plt.show()
