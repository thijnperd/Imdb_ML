import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import pandas as pd
import numpy as np
import joblib
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from tensorflow.keras.models import load_model
import os
from playsound import playsound

# === Load models and tools ===
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
bert_model.eval()

preprocessor = joblib.load("preprocessor.joblib")
scaler = joblib.load("scaler.joblib")
model = load_model("neural_network_model_huber.keras")

# === Load and map country codes ===
df_countries = pd.read_csv("imdb_movies_schoon.csv")
raw_codes = sorted(df_countries['country'].dropna().unique().tolist())

code_to_name = {
    "AR": "Argentina", "AT": "Austria", "AU": "Australia", "BE": "Belgium", "BO": "Bolivia",
    "BR": "Brazil", "BY": "Belarus", "CA": "Canada", "CH": "Switzerland", "CL": "Chile",
    "CN": "China", "CO": "Colombia", "CZ": "Czech Republic", "DE": "Germany", "DK": "Denmark",
    "DO": "Dominican Republic", "ES": "Spain", "FI": "Finland", "FR": "France", "GB": "United Kingdom",
    "GR": "Greece", "GT": "Guatemala", "HK": "Hong Kong", "HU": "Hungary", "ID": "Indonesia",
    "IE": "Ireland", "IL": "Israel", "IN": "India", "IR": "Iran", "IS": "Iceland",
    "IT": "Italy", "JP": "Japan", "KH": "Cambodia", "KR": "South Korea", "LV": "Latvia",
    "MU": "Mauritius", "MX": "Mexico", "MY": "Malaysia", "NL": "Netherlands", "NO": "Norway",
    "PE": "Peru", "PH": "Philippines", "PL": "Poland", "PR": "Puerto Rico", "PT": "Portugal",
    "PY": "Paraguay", "RU": "Russia", "SE": "Sweden", "SG": "Singapore", "SK": "Slovakia",
    "SU": "Soviet Union", "TH": "Thailand", "TR": "Turkey", "TW": "Taiwan", "UA": "Ukraine",
    "US": "United States of America", "UY": "Uruguay", "VN": "Vietnam", "XC": "East Germany",
    "ZA": "South Africa"
}

country_display = [f"{code} - {code_to_name.get(code, 'Unknown')}" for code in raw_codes]

# === BERT function ===
def get_bert_embedding(text):
    tokens = tokenizer(text, padding="max_length", truncation=True,
                       max_length=128, return_tensors='pt')
    with torch.no_grad():
        outputs = bert_model(**tokens)
        cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
    return cls_embedding

# === Prediction function ===
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

        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        df['release_month'] = df['release_date'].dt.month
        df['overview'] = df['overview'].fillna('')
        df['genre'] = df['genre'].fillna('')
        df['cast'] = ''
        df['country'] = df['country'].fillna('')

        X_struct = preprocessor.transform(df)
        X_struct = scaler.transform(X_struct)
        X_bert = get_bert_embedding(overview)

        X_final = np.hstack([X_struct.toarray(), X_bert])

        # Zorg dat X_final exact 1852 features heeft
        expected_dim = 1852
        current_dim = X_final.shape[1]

        if current_dim < expected_dim:
            padding = np.zeros((1, expected_dim - current_dim))
            X_final = np.hstack([X_final, padding])
        elif current_dim > expected_dim:
            X_final = X_final[:, :expected_dim]

        prediction = model.predict(X_final)[0][0] * 100

        if current_dim < expected_dim:
            padding = np.zeros((1, expected_dim - current_dim))
            X_final = np.hstack([X_final, padding])
        elif current_dim > expected_dim:
            X_final = X_final[:, :expected_dim]

        prediction = model.predict(X_final)[0][0] * 100

        return round(prediction, 2)

    except Exception as e:
        return f"Error: {str(e)}"

# === GUI Setup ===
root = tk.Tk()
root.title("Movie Rating Predictor")
root.geometry("700x600")
root.configure(bg="#f0f2f5")
root.resizable(False, False)

style = ttk.Style()
style.theme_use("clam")
style.configure("TFrame", background="#f0f2f5")
style.configure("TLabel", background="#f0f2f5", font=("Segoe UI", 10))
style.configure("TButton", font=("Segoe UI", 10, "bold"), foreground="white", background="#007acc")
style.map("TButton", background=[("active", "#005f99")])
style.configure("TEntry", font=("Segoe UI", 10))
style.configure("TCombobox", font=("Segoe UI", 10))

main_frame = ttk.Frame(root, padding=25, style="TFrame")
main_frame.place(relx=0.5, rely=0.5, anchor="center")

ttk.Label(main_frame, text="Movie Rating Predictor", font=("Segoe UI", 14, "bold")).grid(row=0, column=0, columnspan=2, pady=(0, 20))

ttk.Label(main_frame, text="Movie Description:").grid(row=1, column=0, columnspan=2, sticky="w")
overview_entry = tk.Text(main_frame, width=70, height=6, font=("Segoe UI", 10), wrap="word")
overview_entry.grid(row=2, column=0, columnspan=2, pady=(0, 15))
overview_entry.tag_configure("center", justify='center')

ttk.Label(main_frame, text="Genre:").grid(row=3, column=0, sticky="w")
genre_entry = ttk.Entry(main_frame, width=42, justify="center")
genre_entry.grid(row=4, column=0, sticky="w", pady=(0, 15))

ttk.Label(main_frame, text="Budget ($):").grid(row=3, column=1, sticky="w")
budget_entry = ttk.Entry(main_frame, width=25, justify="center")
budget_entry.insert(0, "10000000")
budget_entry.grid(row=4, column=1, sticky="w", pady=(0, 15))

ttk.Label(main_frame, text="Production Country:").grid(row=5, column=0, sticky="w")
country_combobox = ttk.Combobox(main_frame, values=country_display, state="readonly", width=40, justify="center")
if country_display:
    country_combobox.set(country_display[0])
country_combobox.grid(row=6, column=0, sticky="w", pady=(0, 15))

ttk.Label(main_frame, text="Release Year:").grid(row=5, column=1, sticky="w")
year_entry = ttk.Entry(main_frame, width=25, justify="center")
year_entry.insert(0, "2023")
year_entry.grid(row=6, column=1, sticky="w", pady=(0, 15))

# === Predict + Music Function ===
def on_predict():
    overview = overview_entry.get("1.0", tk.END).strip()
    genre = genre_entry.get().strip()
    budget = budget_entry.get().strip()
    country = country_combobox.get().split(" - ")[0].strip()
    year = year_entry.get().strip()

    if not overview:
        messagebox.showerror("Error", "Please enter a description.")
        return

    overview_entry.tag_add("center", "1.0", "end")
    result = predict_rating(overview, genre, budget, country, year)

    try:
        rating = float(result)
    except:
        messagebox.showerror("Error", f"Invalid prediction result: {result}")
        return

    # Kies liedje op basis van rating (1–10)
    song_index = min(max(int(rating // 10) + 1, 1), 10)
    song_path = os.path.join("bangerwanger", f"{song_index}.mp3")

    if os.path.exists(song_path):
        try:
            playsound(song_path)
        except Exception as e:
            messagebox.showwarning("Warning", f"Kon liedje niet afspelen:\n{e}")
    else:
        messagebox.showwarning("Warning", f"Liedje '{song_path}' bestaat niet.")

    messagebox.showinfo("Prediction", f"Estimated rating: {rating}\nPlaying song {song_index}/10")

predict_btn = ttk.Button(main_frame, text="Predict Rating", command=on_predict)
predict_btn.grid(row=7, column=0, columnspan=2, pady=25)

root.mainloop()
