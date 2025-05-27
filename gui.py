import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
import joblib
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from tensorflow.keras.models import load_model

# === Laad alles ===
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
bert_model.eval()

preprocessor = joblib.load("preprocessor.joblib")
scaler = joblib.load("scaler.joblib")
model = load_model("neural_network_model_huber.keras")

def get_bert_embedding(text):
    tokens = tokenizer(text, padding="max_length", truncation=True,
                       max_length=128, return_tensors='pt')
    with torch.no_grad():
        outputs = bert_model(**tokens)
        cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
    return cls_embedding

def predict_rating(overview, genre, budget, country, year):
    try:
        df = pd.DataFrame([{
            "overview": overview,
            "genre": genre,
            "budget": float(budget),
            "country": country,
            "release_year": int(year),
            "release_date": f"{int(year)}-01-01"
        }])

        # Preprocessing
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        df['release_month'] = df['release_date'].dt.month
        df['overview'] = df['overview'].fillna('')
        df['genre'] = df['genre'].fillna('')
        df['cast'] = ''  # Dummy (leeg) indien nodig
        df['country'] = df['country'].fillna('')

        X_struct = preprocessor.transform(df)
        X_struct = scaler.transform(X_struct)
        X_bert = get_bert_embedding(overview)

        X_final = np.hstack([X_struct.toarray(), X_bert])
        prediction = model.predict(X_final)[0][0] * 100

        return round(prediction, 2)

    except Exception as e:
        return f"Fout: {str(e)}"

# === GUI ===
root = tk.Tk()
root.title("Film Rating Voorspeller")

tk.Label(root, text="Beschrijving:").grid(row=0, sticky="w")
overview_entry = tk.Text(root, width=60, height=6)
overview_entry.grid(row=1, padx=10)

tk.Label(root, text="Genre:").grid(row=2, sticky="w")
genre_entry = tk.Entry(root, width=40)
genre_entry.grid(row=3, padx=10, pady=(0,10))

tk.Label(root, text="Budget:").grid(row=4, sticky="w")
budget_entry = tk.Entry(root, width=40)
budget_entry.insert(0, "10000000")  # default
budget_entry.grid(row=5, padx=10, pady=(0,10))

tk.Label(root, text="Land:").grid(row=6, sticky="w")
country_entry = tk.Entry(root, width=40)
country_entry.insert(0, "United States of America")
country_entry.grid(row=7, padx=10, pady=(0,10))

tk.Label(root, text="Releasejaar:").grid(row=8, sticky="w")
year_entry = tk.Entry(root, width=40)
year_entry.insert(0, "2023")
year_entry.grid(row=9, padx=10, pady=(0,10))

def on_predict():
    overview = overview_entry.get("1.0", tk.END).strip()
    genre = genre_entry.get().strip()
    budget = budget_entry.get().strip()
    country = country_entry.get().strip()
    year = year_entry.get().strip()

    if not overview:
        messagebox.showerror("Fout", "Voer een beschrijving in.")
        return

    result = predict_rating(overview, genre, budget, country, year)
    messagebox.showinfo("Voorspelling", f"Geschatte rating: {result}")

tk.Button(root, text="Voorspel rating", command=on_predict).grid(row=10, pady=10)

root.mainloop()
