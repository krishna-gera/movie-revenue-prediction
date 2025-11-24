import pandas as pd
import numpy as np
import ast
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# --- Configuration ---
MODEL_FILE = 'movie_revenue_model.pkl'
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
# Reduce questions to just the top 5 most impactful features
TOP_N_FEATURES = 5 

FILES = {
    'metadata': 'movies_metadata.csv',
    'movies': 'movies.csv',
    'forbes': 'forbes_celebrity_100.csv',
    'tmdb': 'tmdb_5000_movies.csv',
    'wiki': 'wiki_movie_plots_deduped.csv',
    'high_gross': 'Highest Holywood Grossing Movies.csv'
}

# --- Helper Functions (Same as before) ---
def parse_json_count(data_str):
    try:
        data_list = ast.literal_eval(data_str)
        if isinstance(data_list, list): return len(data_list)
    except: pass
    return 0

def get_file_path(filename):
    return os.path.join(DATA_DIR, filename)

# --- Data Pipeline (Same loading logic) ---
def load_and_merge_data():
    print("\n--- 1. Data Loading & Merging ---")
    path = get_file_path(FILES['metadata'])
    if not os.path.exists(path):
        print(f"CRITICAL ERROR: {FILES['metadata']} not found.")
        return None
    
    print(f"Loading {FILES['metadata']}...")
    df = pd.read_csv(path, low_memory=False)
    df['original_title'] = df['original_title'].astype(str).str.strip()
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['year'] = df['release_date'].dt.year

    # Enrich with Movies.csv
    path = get_file_path(FILES['movies'])
    if os.path.exists(path):
        print("Merging IMDb data...")
        try:
            df_mov = pd.read_csv(path, encoding='latin-1')
            df_mov = df_mov.rename(columns={'name': 'original_title', 'score': 'imdb_score', 'star': 'lead_actor'})
            df = pd.merge(df, df_mov[['original_title', 'year', 'imdb_score', 'lead_actor', 'votes']], 
                          on=['original_title', 'year'], how='left')
        except: pass

    # Enrich with Forbes
    path = get_file_path(FILES['forbes'])
    if os.path.exists(path) and 'lead_actor' in df.columns:
        print("Calculating Star Power...")
        try:
            df_forbes = pd.read_csv(path)
            top_stars = set(df_forbes['Name'].astype(str).str.strip().unique())
            df['has_top_star'] = df['lead_actor'].apply(lambda x: 1 if str(x).strip() in top_stars else 0)
        except: df['has_top_star'] = 0
    else: df['has_top_star'] = 0

    # Enrich with TMDB
    path = get_file_path(FILES['tmdb'])
    if os.path.exists(path):
        print("Merging Popularity metrics...")
        try:
            df_tmdb = pd.read_csv(path)
            df_tmdb = df_tmdb.rename(columns={'original_title': 'original_title', 'popularity': 'tmdb_popularity'})
            df_tmdb = df_tmdb.drop_duplicates(subset=['original_title'])
            df = pd.merge(df, df_tmdb[['original_title', 'tmdb_popularity']], on='original_title', how='left')
        except: pass

    # Enrich with Wiki
    path = get_file_path(FILES['wiki'])
    if os.path.exists(path):
        print("Extracting Plot Complexity...")
        try:
            df_wiki = pd.read_csv(path)
            df_wiki['plot_word_count'] = df_wiki['Plot'].astype(str).apply(lambda x: len(x.split()))
            df_wiki = df_wiki.rename(columns={'Title': 'original_title', 'Release Year': 'year'})
            df_wiki = df_wiki.drop_duplicates(subset=['original_title', 'year'])
            df = pd.merge(df, df_wiki[['original_title', 'year', 'plot_word_count']], on=['original_title', 'year'], how='left')
        except: pass

    # Enrich with High Grossing
    path = get_file_path(FILES['high_gross'])
    if os.path.exists(path):
        print("Checking Historic Blockbusters...")
        try:
            df_high = pd.read_csv(path)
            blockbusters = set(df_high['Title'].astype(str).str.strip())
            df['is_historic_blockbuster'] = df['original_title'].apply(lambda x: 1 if x in blockbusters else 0)
        except: df['is_historic_blockbuster'] = 0
    else: df['is_historic_blockbuster'] = 0

    return df

def preprocess_data(df):
    print("\n--- 2. Preprocessing ---")
    if 'production_companies' in df.columns:
        df['num_production_companies'] = df['production_companies'].apply(parse_json_count)
    if 'production_countries' in df.columns:
        df['num_production_countries'] = df['production_countries'].apply(parse_json_count)

    for col in ['budget', 'revenue', 'runtime']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill missing with median/0
    defaults = {
        'imdb_score': df['imdb_score'].median() if 'imdb_score' in df else 6.0,
        'votes': 0, 'tmdb_popularity': 0, 'plot_word_count': 300,
        'has_top_star': 0, 'is_historic_blockbuster': 0
    }
    for col, val in defaults.items():
        if col in df.columns: df[col] = df[col].fillna(val)
    
    df['release_month'] = df['release_date'].dt.month.fillna(6)
    df = df.fillna(0)
    df = df[(df['budget'] > 1000) & (df['revenue'] > 1000)]
    
    features = [
        'budget', 'runtime', 'num_production_companies', 
        'num_production_countries', 'release_month', 
        'imdb_score', 'votes', 'tmdb_popularity', 
        'has_top_star', 'is_historic_blockbuster', 'plot_word_count',
        'revenue'
    ]
    return df[[c for c in features if c in df.columns]]

def remove_outliers(df):
    print("Removing outliers...")
    for col in ['budget', 'revenue', 'runtime']:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df = df[(df[col] >= (Q1 - 1.5 * IQR)) & (df[col] <= (Q3 + 1.5 * IQR))]
    return df

def train_and_save():
    df = load_and_merge_data()
    if df is None: return None
    
    df_clean = preprocess_data(df)
    df_final = remove_outliers(df_clean)
    
    X = df_final.drop(columns=['revenue'])
    y = df_final['revenue']
    
    # Calculate feature importance to select top questions
    # We normalize data just for selection to be fair, but train on raw
    # Simple approach: Train model, check coefficients magnitude*std_dev
    model = LinearRegression()
    model.fit(X, y)
    
    # Determine "Important" features (Impact = Coef * Std Dev of feature)
    # This tells us which features actually move the needle most often
    feature_impact = {}
    for feat, coef in zip(X.columns, model.coef_):
        impact = abs(coef * X[feat].std())
        feature_impact[feat] = impact
        
    sorted_features = sorted(feature_impact.items(), key=lambda x: x[1], reverse=True)
    top_features = [f[0] for f in sorted_features[:TOP_N_FEATURES]]
    
    print(f"\nTop {TOP_N_FEATURES} impactful features selected for user input: {top_features}")
    
    # Save everything including defaults (means) for the skipped questions
    feature_defaults = X.mean().to_dict()
    
    model_package = {
        'model': model,
        'features': X.columns.tolist(),
        'top_features': top_features,
        'defaults': feature_defaults
    }
    joblib.dump(model_package, os.path.join(DATA_DIR, MODEL_FILE))
    print(f"Model saved to {MODEL_FILE}")
    return model_package

def visualize_prediction(base, impacts, final_pred):
    """Generates a Waterfall chart of the prediction logic."""
    # Data for plotting
    labels = ['Base'] + [x[0] for x in impacts] + ['Prediction']
    values = [base] + [x[1] for x in impacts] + [final_pred]
    
    # Re-sorting impacts for graph
    impact_names = [x[0] for x in impacts]
    impact_vals = [x[1] for x in impacts]
    
    plt.figure(figsize=(10, 6))
    plt.barh(impact_names, impact_vals, color=['green' if x > 0 else 'red' for x in impact_vals])
    plt.title(f'Revenue Drivers (Predicted: ${final_pred:,.0f})')
    plt.xlabel('Impact on Revenue ($)')
    plt.axvline(x=0, color='black', linestyle='-')
    plt.tight_layout()
    
    print("\n[Graph generated. Check popup window]")
    plt.show()

def predict_revenue(model_pkg):
    model = model_pkg['model']
    features = model_pkg['features']
    top_feats = model_pkg['top_features']
    defaults = model_pkg['defaults']
    
    print("\n" + "="*40)
    print("   SMART MOVIE REVENUE PREDICTOR   ")
    print("="*40)
    print(f"I will only ask about the {len(top_feats)} most important factors.")
    print("Everything else will use industry averages.")
    
    user_inputs = defaults.copy()
    
    for feat in top_feats:
        try:
            default_val = defaults.get(feat, 0)
            
            if feat == 'has_top_star':
                val = input("Is lead actor a Forbes Top Star? (1=Yes, 0=No) [0]: ")
            elif feat == 'is_historic_blockbuster':
                val = input("Is it a historic blockbuster franchise? (1=Yes, 0=No) [0]: ")
            elif feat == 'budget':
                val = input(f"Budget ($) [Avg: ${default_val:,.0f}]: ")
            else:
                clean_name = feat.replace('_', ' ').title()
                if feat == 'imdb_score': val = input(f"{clean_name} (0-10) [Avg: {default_val:.1f}]: ")
                else: val = input(f"{clean_name} [Avg: {default_val:.1f}]: ")
            
            if val.strip() != "":
                user_inputs[feat] = float(val)
                
        except ValueError:
            print("Invalid. Using average.")

    # Prepare dataframe in correct column order
    input_df = pd.DataFrame([user_inputs], columns=features)
    prediction = max(0, model.predict(input_df)[0])
    
    print("\n" + "-"*40)
    print(f"PREDICTED REVENUE: ${prediction:,.2f}")
    print("-" * 40)
    
    # Analyze impacts for the graph
    impacts = []
    # Only show impacts for the top features we asked about + top hidden ones if large
    for feat, coef in zip(features, model.coef_):
        val = user_inputs[feat]
        impact = val * coef
        # Filter for significant impact to keep graph readable
        if abs(impact) > 1_000_000: 
            impacts.append((feat, impact))
            
    # Sort by magnitude
    impacts.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # Show Graph
    visualize_prediction(model.intercept_, impacts, prediction)

if __name__ == "__main__":
    model_path = os.path.join(DATA_DIR, MODEL_FILE)
    
    model_data = None
    
    if os.path.exists(model_path):
        print(f"Found existing model: {MODEL_FILE}")
        try:
            model_data = joblib.load(model_path)
            # Verify this is the NEW model format with 'top_features'
            if 'top_features' not in model_data:
                print("Old model format detected. Retraining to upgrade...")
                model_data = train_and_save()
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Retraining...")
            model_data = train_and_save()
    else:
        print("Model not found. Initializing training...")
        model_data = train_and_save()
    
    if model_data:
        while True:
            predict_revenue(model_data)
            if input("\nPredict another? (y/n): ").lower() != 'y': break