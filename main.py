
# %% [markdown]

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input


DATA_PATH = 'data/IRIS.csv'
OUTPUT_PATH = 'output/'
os.makedirs(OUTPUT_PATH, exist_ok=True)
sns.set(style="whitegrid")

print(f"TensorFlow Version: {tf.__version__}")

# %% [markdown]

# %%
# Load Data
try:
    df = pd.read_csv(DATA_PATH)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    sys.exit(f"Critical Error: {DATA_PATH} not found. Did you move the Kaggle csv file there?")

df.columns = df.columns.str.lower()

if 'id' in df.columns:
    df = df.drop(columns=['id'])

print("Columns found:", df.columns.tolist())
print(df.head())


sns.pairplot(df, hue="species", markers=["o", "s", "D"])
plt.suptitle("Visualisasi Fitur Iris", y=1.02)
plt.savefig(f"{OUTPUT_PATH}data_visualization.png")
plt.show()


target_col = 'species' if 'species' in df.columns else 'Species'
encoder = LabelEncoder()
y = encoder.fit_transform(df[target_col])
y_cat = to_categorical(y)

# Scaling Features
X = df.drop(target_col, axis=1).values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split Data (80:20)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_cat, test_size=0.2, random_state=42)
print(f"Training shape: {X_train.shape}, Testing shape: {X_test.shape}")

# %% [markdown]


# %%
model = Sequential([
    Input(shape=(4,)),
    Dense(16, activation='relu', name='Hidden_1'),
    Dense(16, activation='relu', name='Hidden_2'),
    Dense(3, activation='softmax', name='Output')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# %% [markdown]
# ## Bab 3: Hasil Eksperimen

# %%
# Training
history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.2, verbose=0)
print("Training selesai.")

# %%
# Plot Loss & Accuracy
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Model Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig(f"{OUTPUT_PATH}training_history.png")
plt.show()

# %%
# Evaluasi & Confusion Matrix
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nFinal Test Accuracy: {acc*100:.2f}%")

y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.title('Confusion Matrix')
plt.savefig(f"{OUTPUT_PATH}confusion_matrix.png")
plt.show()
