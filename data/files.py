import pandas as pd

print("Creating sample datasets...")

# Load and sample movies_metadata.csv
try:
    df = pd.read_csv('movies_metadata.csv', low_memory=False)
    print(f"Original movies_metadata.csv: {len(df)} rows")
    df_sample = df.head(5000)
    df_sample.to_csv('data/movies_metadata_sample.csv', index=False)
    print(f"Created movies_metadata_sample.csv: {len(df_sample)} rows")
except Exception as e:
    print(f"Error with movies_metadata.csv: {e}")

# Load and sample credits.csv
try:
    df = pd.read_csv('credits.csv')
    print(f"Original credits.csv: {len(df)} rows")
    df_sample = df.head(5000)
    df_sample.to_csv('data/credits_sample.csv', index=False)
    print(f"Created credits_sample.csv: {len(df_sample)} rows")
except Exception as e:
    print(f"Error with credits.csv: {e}")

# Load and sample keywords.csv
try:
    df = pd.read_csv('keywords.csv')
    print(f"Original keywords.csv: {len(df)} rows")
    df_sample = df.head(5000)
    df_sample.to_csv('data/keywords_sample.csv', index=False)
    print(f"Created keywords_sample.csv: {len(df_sample)} rows")
except Exception as e:
    print(f"Error with keywords.csv: {e}")

print("\n✅ Sample files created!")
print("Now update app.py to use these sample files")