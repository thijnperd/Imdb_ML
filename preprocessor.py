import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import joblib

# Functie die een DataFrame voorbereidt op wat de preprocessor verwacht
def prepare_dataframe(df):
    df.columns = df.columns.str.strip()

    # Zorg dat 'release_date' een datetime is
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

    # Voeg maand en jaar toe
    df['release_month'] = df['release_date'].dt.month
    df['release_year'] = df['release_date'].dt.year

    # Vul ontbrekende tekstvelden aan met lege string
    for col in ['overview', 'genre', 'cast']:
        df[col] = df[col].fillna('')

    return df

# === Stap 1: Data inladen ===
df = pd.read_csv("imdb_movies_schoon.csv", skipinitialspace=True)

# === Stap 2: Voorbewerken ===
df = prepare_dataframe(df)

# === Stap 3: Preprocessor instellen ===
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), ["budget", "release_month", "release_year"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["country"]),
        ("genre_text", TfidfVectorizer(max_features=200), "genre"),
        ("cast_text", TfidfVectorizer(max_features=1000), "cast"),
    ]
)

# === Stap 4: Preprocessor trainen en opslaan ===
preprocessor.fit(df)
joblib.dump(preprocessor, "preprocessor.joblib")

print("Preprocessor is opgeslagen als 'preprocessor.joblib'.")
print("Details van de preprocessor:")
for name, transformer, columns in preprocessor.transformers:
    print(f"- Transformer '{name}': {transformer.__class__.__name__} op kolom(men): {columns}")
