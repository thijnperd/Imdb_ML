import pandas as pd
import joblib

# CSV-bestand inladen
df = pd.read_csv("imdb_movies_schoon.csv", skipinitialspace=True)
df.columns = df.columns.str.strip()

# Voeg 'release_month' en 'release_year' toe op basis van de 'release_date'
df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
df["release_month"] = df["release_date"].dt.month
df["release_year"] = df["release_date"].dt.year

# Laad de preprocessor die je eerder hebt opgeslagen
preprocessor = joblib.load("preprocessor.joblib")

# Test de preprocessor op een subset van de dataset (gebruik de eerste paar rijen)
X_test = df.head()  # Gebruik de eerste paar rijen om snel te testen
X_transformed = preprocessor.transform(X_test)

# Bekijk de getransformeerde data (voorbeeld van de eerste paar rijen)
print("Getransformeerde data (voorbeeld):")
print(X_transformed.toarray())  # Zet het om naar een array om het beter te kunnen bekijken

# Voor OneHotEncoder (categorieën van 'country')
country_encoder = preprocessor.transformers_[1][1]
print("\nOne-hot encoding van 'country':")
print(country_encoder.categories_)

# Voor TfidfVectorizer (overview, genre, cast)
overview_vectorizer = preprocessor.transformers_[2][1]
print("\nKenmerken van de TfidfVectorizer voor 'overview':")
print(overview_vectorizer.get_feature_names_out())

genre_vectorizer = preprocessor.transformers_[3][1]
print("\nKenmerken van de TfidfVectorizer voor 'genre':")
print(genre_vectorizer.get_feature_names_out())

cast_vectorizer = preprocessor.transformers_[4][1]
print("\nKenmerken van de TfidfVectorizer voor 'cast':")
print(cast_vectorizer.get_feature_names_out())

# Test de geladen preprocessor opnieuw
X_transformed_loaded = preprocessor.transform(X_test)
print("\nGetransformeerde data na herladen:")
print(X_transformed_loaded.toarray())
