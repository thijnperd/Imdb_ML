import matplotlib.pyplot as plt
import pandas as pd
import joblib
import numpy as np
import tensorflow as tf
import time
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import MeanSquaredError  # Aangepast naar MSE
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from transformers import DistilBertTokenizer, DistilBertModel
import torch

# === BERT-initialisatie ===
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
bert_model.eval()

def get_bert_embeddings(text_series, max_len=128, batch_size=32):
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(text_series), batch_size):
            batch = text_series[i:i+batch_size].tolist()
            tokens = tokenizer(batch, padding="max_length", truncation=True,
                               max_length=max_len, return_tensors='pt')
            outputs = bert_model(**tokens)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_embeddings)
    return np.vstack(embeddings)

def load_or_generate_bert_embeddings(name_prefix, text_series):
    emb_path = f"{name_prefix}_bert.joblib"
    if os.path.exists(emb_path):
        print(f"📂 BERT-embeddings geladen vanaf: {emb_path}")
        return joblib.load(emb_path)
    else:
        print(f"🧠 BERT-embeddings genereren voor: {name_prefix}")
        embeddings = get_bert_embeddings(text_series)
        joblib.dump(embeddings, emb_path)
        return embeddings

def prepare_dataframe(df):
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['release_month'] = df['release_date'].dt.month
    df['release_year'] = df['release_date'].dt.year
    for col in ['overview', 'genre', 'cast']:
        df[col] = df[col].fillna('')
    return df

# === Data inladen en splitsen ===
df = pd.read_csv("imdb_movies_schoon.csv", skipinitialspace=True)
df.columns = df.columns.str.strip()
train_val, test = train_test_split(df, test_size=0.2, random_state=42)
train, val = train_test_split(train_val, test_size=0.2, random_state=42)
train = prepare_dataframe(train)
val = prepare_dataframe(val)
test = prepare_dataframe(test)

# === Preprocessing ===
preprocessor = joblib.load("preprocessor.joblib")
X_train_struct = preprocessor.transform(train)
X_val_struct = preprocessor.transform(val)
X_test_struct = preprocessor.transform(test)
scaler = StandardScaler(with_mean=False)
X_train_struct = scaler.fit_transform(X_train_struct)
X_val_struct = scaler.transform(X_val_struct)
X_test_struct = scaler.transform(X_test_struct)

# === BERT embeddings ===
X_train_bert = load_or_generate_bert_embeddings("X_train", train["overview"])
X_val_bert = load_or_generate_bert_embeddings("X_val", val["overview"])
X_test_bert = load_or_generate_bert_embeddings("X_test", test["overview"])

# === Combineren ===
X_train = np.hstack([X_train_struct.toarray(), X_train_bert])
X_val = np.hstack([X_val_struct.toarray(), X_val_bert])
X_test = np.hstack([X_test_struct.toarray(), X_test_bert])

# === Genormaliseerde target instellen ===
y_train = train['rating'].astype(float).values / 100.0
y_val = val['rating'].astype(float).values / 100.0
y_test = test['rating'].astype(float).values / 100.0

# === Model bouwen ===
model = Sequential([
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.3),
    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.3),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss=MeanSquaredError(),  # Veranderd naar MSE
              metrics=['mae'])

model.summary()

# === Early stopping ===
early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)

# === Training ===
start_time = time.time()
history = model.fit(
    X_train, y_train,
    epochs=500,
    batch_size=50,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],
    verbose=1
)
training_time = time.time() - start_time

# === Evaluatie (met terugschalen MAE) ===
def evaluate_and_rescale(model, X, y_true, label):
    loss, mae = model.evaluate(X, y_true, verbose=0)
    scaled_mae = mae * 100
    scaled_mse = loss * 100**2
    print(f"--- {label.upper()} ---\nLoss (MSE): {scaled_mse:.4f}\nMAE: {scaled_mae:.4f}")
    return scaled_mse, scaled_mae

train_loss, train_mae = evaluate_and_rescale(model, X_train, y_train, "train")
val_loss, val_mae = evaluate_and_rescale(model, X_val, y_val, "validatie")
test_loss, test_mae = evaluate_and_rescale(model, X_test, y_test, "test")

# === Opslaan resultaten ===
result_text = f"""
🔹 Neural Network Evaluatieoverzicht (Regressie):
📊 Model parameters: {model.count_params()}
⏱️ Trainingstijd: {training_time:.2f} seconden

--- TRAIN ---
Loss (MSE)      : {train_loss:.4f}
MAE             : {train_mae:.4f}

--- VALIDATIE ---
Loss (MSE)      : {val_loss:.4f}
MAE             : {val_mae:.4f}

--- TEST ---
Loss (MSE)      : {test_loss:.4f}
MAE             : {test_mae:.4f}
"""

with open("result.txt", "w", encoding="utf-8") as f:
    f.write(result_text.strip())

print(result_text)

# === Visualisatie ===
plt.plot(history.history['loss'], label='Training loss (MSE)')
plt.plot(history.history['val_loss'], label='Validation loss (MSE)')
plt.title('Training en Validatieverlies (MSE)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# === Opslaan model ===
model.save("neural_network_model_MSE.keras")
print("✅ Model opgeslagen als 'neural_network_model_MSE.keras'.")
