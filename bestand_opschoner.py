import pandas as pd

# CSV-bestand inladen
df = pd.read_csv("imdb_movies.csv", skipinitialspace=True)
df.columns = df.columns.str.strip()

# Verwijder dubbele rijen
df = df.drop_duplicates()

# Controleer de eerste paar rijen van het bestand om te zien of het goed is ingeladen
print("Eerste paar rijen van de originele dataset (zonder duplicaten):")
print(df.head())

# Hernoem kolommen
df = df.rename(columns={
    "date_x": "release_date",
    "score": "rating",
    "budget_x": "budget"
})

# Verwijder ongewenste kolommen
df = df.drop(columns=["names", "orig_title", "status", "orig_lang"], errors="ignore")

# Controleer of er lege waarden zijn in de kolommen die we willen behouden
print("\nAantal ontbrekende waarden per kolom na het hernoemen van kolommen:")
print(df.isnull().sum())

# Converteer release_date naar datetime (automatische herkenning van formaat)
df["release_date"] = pd.to_datetime(df["release_date"], dayfirst=False, errors="coerce")

# Controleer of de conversie gelukt is
print("\nVoorbeeld van geconverteerde datums:")
print(df["release_date"].head())

# Verwijder rijen met ongeldige datums
df = df.dropna(subset=["release_date"])

# Controleer opnieuw de dataset na het verwijderen van NaT waarden
print("\nAantal rijen na het verwijderen van ongeldige datums:")
print(df.shape[0])

# Eventueel strings schoonmaken in 'genre' en 'cast' (als die kolommen bestaan)
if "genre" in df.columns:
    df["genre"] = df["genre"].astype(str).str.replace(",", " ", regex=False)
if "cast" in df.columns:
    df["cast"] = df["cast"].astype(str).str.replace(",", " ", regex=False)

# Sla het schone bestand op
df.to_csv("imdb_movies_schoon.csv", index=False)

print("\nHet bestand is opgeschoond en opgeslagen als 'imdb_movies_schoon.csv'.")
