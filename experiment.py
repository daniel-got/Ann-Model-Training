# experiment.py - Khusus untuk Tuning Hyperparameter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Library Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

DATA_PATH = 'data/IRIS.csv'
OUTPUT_PATH = 'output/'
os.makedirs(OUTPUT_PATH, exist_ok=True)
sns.set(style="whitegrid")

print(">>> Memulai Script Eksperimen...")

# ==========================================
# BAGIAN 1: PREPROCESSING
# ==========================================
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    sys.exit(f"Error: {DATA_PATH} tidak ditemukan.")

# Standarisasi Kolom
df.columns = df.columns.str.lower()
if 'id' in df.columns: df = df.drop(columns=['id'])

# Encoding & Scaling
encoder = LabelEncoder()
y = to_categorical(encoder.fit_transform(df['species']))
X = df.drop('species', axis=1).values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ==========================================
# BAGIAN 2: GRID SEARCH LOOP (LR vs Epochs)
# ==========================================
learning_rates = [0.01, 0.001, 0.0001]
epochs_list = [30, 70, 100]

results = []
plt.figure(figsize=(18, 12))
plot_idx = 1

print(f">>> Testing kombinasi {len(learning_rates)} LR x {len(epochs_list)} Epochs...")

for lr in learning_rates:
    for epoch_target in epochs_list:
        print(f"   -> Training: LR={lr}, Epochs={epoch_target}...")
        
        # Reset Model setiap iterasi
        model = Sequential([
            Input(shape=(4,)),
            Dense(16, activation='relu'),
            Dense(16, activation='relu'),
            Dense(3, activation='softmax')
        ])
        
        # Compile dengan LR dinamis
        opt = Adam(learning_rate=lr)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        
        # Training
        history = model.fit(X_train, y_train, epochs=epoch_target, 
                            batch_size=16, validation_split=0.2, verbose=0)
        
        # Evaluasi
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        results.append({
            'Learning Rate': lr,
            'Epochs': epoch_target,
            'Accuracy': f"{accuracy*100:.2f}%",
            'Final Loss': f"{loss:.4f}"
        })
        
        # Plotting
        plt.subplot(3, 3, plot_idx)
        plt.plot(history.history['loss'], label='Train', linewidth=2)
        plt.plot(history.history['val_loss'], label='Val', linestyle='--')
        plt.title(f'LR: {lr} | Epochs: {epoch_target}\nTest Acc: {accuracy*100:.1f}%')
        
        if plot_idx > 6: plt.xlabel('Epoch')
        if plot_idx % 3 == 1: plt.ylabel('Loss')
        
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plot_idx += 1

plt.tight_layout()
plt.suptitle("Hyperparameter Tuning Results (LR vs Epochs)", y=1.02, fontsize=16)

# Simpan Gambar Eksperimen
save_file = OUTPUT_PATH + 'grid_search_experiment.png'
plt.savefig(save_file, bbox_inches='tight')
print(f"\n>>> Grafik disimpan di: {save_file}")

# Tampilkan Tabel
print("\n=== REKAP HASIL ===")
print(pd.DataFrame(results))
