import pandas as pd
import numpy as np
import ast
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# --- Configuration ---
MODEL_FILE = 'movie_revenue_model.pkl'
DATA_DIR = 'data'

FILES = {
    'metadata': 'movies_metadata.csv',
    'movies': 'movies.csv',
    'forbes': 'forbes_celebrity_100.csv',
    'tmdb': 'tmdb_5000_movies.csv',
    'wiki': 'wiki_movie_plots_deduped.csv',
    'high_gross': 'Highest Holywood Grossing Movies.csv'
}


def parse_json_count(data_str):
    """Return number of items in a JSON-like list string."""
    try:
        data_list = ast.literal_eval(data_str)
        if isinstance(data_list, list):
            return len(data_list)
    except Exception:
        pass
    return 0


def parse_json_first(data_str, key='name'):
    """Return first element's key from a JSON-like list string."""
    try:
        data_list = ast.literal_eval(data_str)
        if isinstance(data_list, list) and len(data_list) > 0:
            value = data_list[0]
            if isinstance(value, dict):
                return value.get(key, 'Unknown')
            return str(value)
    except Exception:
        pass
    return 'Unknown'


def get_file_path(filename):
    return os.path.join(DATA_DIR, filename)


def load_and_merge_data():
    print("\n--- 1. Data Loading & Merging ---")
    path = get_file_path(FILES['metadata'])
    if not os.path.exists(path):
        print(f"CRITICAL ERROR: {FILES['metadata']} not found in {DATA_DIR}/")
        return None

    print(f"Loading {FILES['metadata']}...")
    df = pd.read_csv(path, low_memory=False)

    # Basic cleaning
    df['original_title'] = df['original_title'].astype(str).str.strip()
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['year'] = df['release_date'].dt.year
    df['release_month'] = df['release_date'].dt.month

    # Extract genre & production_company from JSON-like fields
    df['genre'] = df['genres'].apply(lambda x: parse_json_first(x, 'name'))
    df['production_company'] = df['production_companies'].apply(lambda x: parse_json_first(x, 'name'))

    # Enrich with Movies.csv (IMDb-style data)
    path = get_file_path(FILES['movies'])
    if os.path.exists(path):
        print("Merging IMDb data...")
        try:
            df_mov = pd.read_csv(path, encoding='latin-1')
            df_mov = df_mov.rename(
                columns={
                    'name': 'original_title',
                    'score': 'imdb_score',
                    'star': 'lead_actor',
                    'director': 'director',
                    'writer': 'writer'
                }
            )
            df = pd.merge(
                df,
                df_mov[['original_title', 'year', 'imdb_score',
                        'lead_actor', 'director', 'writer', 'votes']],
                on=['original_title', 'year'],
                how='left'
            )
        except Exception as e:
            print("Error merging Movies.csv:", e)

    # Enrich with Forbes (Star Power)
    path = get_file_path(FILES['forbes'])
    if os.path.exists(path) and 'lead_actor' in df.columns:
        print("Calculating Star Power...")
        try:
            df_forbes = pd.read_csv(path)
            top_stars = set(df_forbes['Name'].astype(str).str.strip().unique())
            df['has_top_star'] = df['lead_actor'].apply(
                lambda x: 1 if str(x).strip() in top_stars else 0
            )
        except Exception as e:
            print("Error merging Forbes data:", e)
            df['has_top_star'] = 0
    else:
        df['has_top_star'] = 0

    # Enrich with TMDB popularity metrics
    path = get_file_path(FILES['tmdb'])
    if os.path.exists(path):
        print("Merging Popularity metrics from TMDB...")
        try:
            df_tmdb = pd.read_csv(path)
            df_tmdb = df_tmdb.drop_duplicates(subset=['original_title'])
            df = pd.merge(
                df,
                df_tmdb[['original_title', 'vote_average', 'vote_count']],
                on='original_title',
                how='left'
            )
        except Exception as e:
            print("Error merging TMDB data:", e)

    return df


def engineer_features(df):
    print("\n--- 2. Feature Engineering ---")

    # Numeric conversions
    for col in ['budget', 'revenue', 'runtime']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Company tier (Major vs Independent)
    major_studios = ['Warner', 'Universal', 'Paramount',
                     'Disney', 'Fox', 'Sony', 'Columbia']
    df['is_major_studio'] = df['production_company'].apply(
        lambda x: 1 if any(studio in str(x) for studio in major_studios) else 0
    )

    # Number of production companies
    if 'production_companies' in df.columns:
        df['num_production_companies'] = df['production_companies'].apply(parse_json_count)
    else:
        df['num_production_companies'] = 1

    # Release season
    def get_season(month):
        if pd.isna(month):
            return 'Unknown'
        month = int(month)
        if month in [5, 6, 7, 8]:
            return 'Summer'
        elif month in [11, 12]:
            return 'Holiday'
        elif month in [1, 2, 3]:
            return 'Awards'
        else:
            return 'Regular'

    df['release_season'] = df['release_month'].apply(get_season)

    # Sequel detection (simple heuristic on title)
    sequel_keywords = ['2', 'II', 'III', 'Part', 'Chapter', 'Returns', 'Resurrection']
    df['is_sequel'] = df['original_title'].apply(
        lambda x: 1 if any(kw in str(x) for kw in sequel_keywords) else 0
    )

    # Fill missing values
    df['genre'] = df['genre'].fillna('Unknown')
    df['imdb_score'] = df['imdb_score'].fillna(
        df['imdb_score'].median() if 'imdb_score' in df else 6.0
    )
    df['runtime'] = df['runtime'].fillna(df['runtime'].median())
    df['release_month'] = df['release_month'].fillna(6)
    df['votes'] = df['votes'].fillna(0) if 'votes' in df.columns else 0
    df['has_top_star'] = df['has_top_star'].fillna(0)
    df['is_major_studio'] = df['is_major_studio'].fillna(0)
    df['num_production_companies'] = df['num_production_companies'].fillna(1)
    df['is_sequel'] = df['is_sequel'].fillna(0)

    # Filter valid rows
    df = df[(df['budget'] > 1000) & (df['revenue'] > 1000)]

    return df


def train_model(df):
    print("\n--- 3. Training Model ---")

    # Select features for training
    feature_columns = [
        'budget', 'runtime', 'release_month', 'imdb_score',
        'num_production_companies', 'is_major_studio', 'has_top_star',
        'is_sequel', 'votes'
    ]

    # Encoders for categorical features
    le_genre = LabelEncoder()
    le_season = LabelEncoder()

    df['genre_encoded'] = le_genre.fit_transform(df['genre'].astype(str))
    df['season_encoded'] = le_season.fit_transform(df['release_season'].astype(str))

    feature_columns.extend(['genre_encoded', 'season_encoded'])

    # Prepare X and y
    X = df[feature_columns].fillna(0)
    y = df['revenue']

    # Remove outliers (IQR)
    Q1 = y.quantile(0.25)
    Q3 = y.quantile(0.75)
    IQR = Q3 - Q1
    mask = (y >= (Q1 - 1.5 * IQR)) & (y <= (Q3 + 1.5 * IQR))
    X = X[mask]
    y = y[mask]

    print(f"Training on {len(X)} samples.")

    # Random Forest model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)

    # Feature importance
    feature_importance = dict(zip(feature_columns, model.feature_importances_))
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

    print("\nTop Feature Importance:")
    for feat, imp in sorted_features[:10]:
        print(f"  {feat}: {imp:.4f}")

    # Default values for features
    feature_defaults = X.median().to_dict()

    # Save model package (IMPORTANT: includes label_encoders)
    model_package = {
        'model': model,
        'features': feature_columns,
        'defaults': feature_defaults,
        'label_encoders': {
            'genre': le_genre,
            'season': le_season
        },
        'genre_classes': list(le_genre.classes_),
        'season_classes': list(le_season.classes_)
    }

    joblib.dump(model_package, MODEL_FILE)
    print(f"\n✓ Model saved to {MODEL_FILE}")
    print("You can now run the Flask app with: python app.py")

    return model_package


if __name__ == "__main__":
    df = load_and_merge_data()
    if df is not None:
        df = engineer_features(df)
        train_model(df)
