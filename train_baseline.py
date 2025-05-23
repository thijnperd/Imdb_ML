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
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from transformers import DistilBertTokenizer, DistilBertModel
import torch

# === 0. BERT-initialisatie ===
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
            cls_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
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

# === 1. Data inladen ===
df = pd.read_csv("imdb_movies_schoon.csv", skipinitialspace=True)
df.columns = df.columns.str.strip()
print("Kolomnamen in df:", df.columns.tolist())

# === 2. Data splitsen in train/val/test ===
train_val, test = train_test_split(df, test_size=0.2, random_state=42)
train, val = train_test_split(train_val, test_size=0.2, random_state=42)

# === 3. Voorbewerking uitvoeren ===
train = prepare_dataframe(train)
val = prepare_dataframe(val)
test = prepare_dataframe(test)

# === 4. Preprocessor laden ===
preprocessor = joblib.load("preprocessor.joblib")  # Dit preprocessor object verwerkt o.a. one-hot encoding

# === 5. Gestructureerde features (zonder 'overview') transformeren ===
X_train_struct = preprocessor.transform(train)
X_val_struct = preprocessor.transform(val)
X_test_struct = preprocessor.transform(test)

# === 6. Normaliseren (zonder mean vanwege sparse matrix) ===
scaler = StandardScaler(with_mean=False)
X_train_struct = scaler.fit_transform(X_train_struct)
X_val_struct = scaler.transform(X_val_struct)
X_test_struct = scaler.transform(X_test_struct)

# === 7. BERT-embeddings voor overview-kolom ===
X_train_bert = load_or_generate_bert_embeddings("X_train", train["overview"])
X_val_bert = load_or_generate_bert_embeddings("X_val", val["overview"])
X_test_bert = load_or_generate_bert_embeddings("X_test", test["overview"])

# === 8. Combineer structured + BERT embeddings ===
X_train = np.hstack([X_train_struct.toarray(), X_train_bert])
X_val = np.hstack([X_val_struct.toarray(), X_val_bert])
X_test = np.hstack([X_test_struct.toarray(), X_test_bert])

print("Kolomnamen in train:", train.columns.tolist())

# === 9. Target-kolom instellen (regressie, dus directe numerieke rating) ===
y_train = train['rating'].astype(float).values
y_val = val['rating'].astype(float).values
y_test = test['rating'].astype(float).values

# === 10. TensorFlow Keras Model bouwen (regressie) ===
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.3))
model.add(Dense(1))  # Output is 1 neuron zonder activatie (lineair)

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='mean_squared_error',
              metrics=['mean_absolute_error'])
model.summary()

# === 11. Early stopping om overfitting te voorkomen ===
early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)

# === 12. Model trainen ===
start_time = time.time()
history = model.fit(X_train, y_train,
                    epochs=50,
                    batch_size=50,
                    validation_data=(X_val, y_val),
                    callbacks=[early_stopping],
                    verbose=1)
training_time = time.time() - start_time

# === 13. Evaluatie ===
train_loss, train_mae = model.evaluate(X_train, y_train, verbose=0)
val_loss, val_mae = model.evaluate(X_val, y_val, verbose=0)
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)

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

# Aangepaste regel met UTF-8 encoding
with open("result.txt", "w", encoding="utf-8") as f:
    f.write(result_text.strip())

# Nieuw toegevoegd: print de evaluatietekst naar console
print(result_text)

# === 14. Visualisatie van training vs validatie verlies ===
plt.plot(history.history['loss'], label='Training loss (MSE)')
plt.plot(history.history['val_loss'], label='Validation loss (MSE)')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.show()

# === 15. Model opslaan ===
model.save("neural_network_model.keras")
print("✅ Neural network model opgeslagen als 'neural_network_model.keras'.")
print("📄 Evaluatieresultaten opgeslagen als 'result.txt'.")
