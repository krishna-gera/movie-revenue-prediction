from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import os
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

app = Flask(__name__)

MODEL_FILE = 'movie_revenue_model.pkl'
model_data = None

# Reference data for dropdowns
GENRES = [
    'Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary',
    'Drama', 'Family', 'Fantasy', 'Horror', 'Mystery', 'Romance',
    'Science Fiction', 'Thriller', 'War', 'Western'
]

MPAA_RATINGS = ['G', 'PG', 'PG-13', 'R', 'NC-17', 'Not Rated']

MAJOR_STUDIOS = [
    'Walt Disney Pictures', 'Warner Bros.', 'Universal Pictures',
    'Paramount Pictures', '20th Century Fox', 'Sony Pictures',
    'Columbia Pictures', 'DreamWorks', 'Lionsgate', 'Independent Studio'
]

KNOWN_DIRECTORS = [
    'Steven Spielberg', 'Christopher Nolan', 'James Cameron',
    'Martin Scorsese', 'Quentin Tarantino', 'Peter Jackson',
    'Ridley Scott', 'Denis Villeneuve', 'Greta Gerwig',
    'Rian Johnson', 'Jordan Peele', 'Other/Unknown'
]

KNOWN_WRITERS = [
    'Aaron Sorkin', 'Quentin Tarantino', 'Christopher Nolan',
    'Greta Gerwig', 'Charlie Kaufman', 'Coen Brothers',
    'Paul Thomas Anderson', 'Other/Unknown'
]

IP_SOURCES = [
    'Original', 'Book Adaptation', 'Comic/Graphic Novel', 'Remake',
    'Video Game', 'TV Series', 'True Story'
]

LANGUAGES = [
    'English', 'Spanish', 'French', 'Mandarin', 'Hindi', 'Japanese',
    'Korean', 'German', 'Italian', 'Portuguese'
]


def load_model():
    global model_data
    if os.path.exists(MODEL_FILE):
        try:
            model_data = joblib.load(MODEL_FILE)
            print("Model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            model_data = None
            return False
    else:
        print("Model file not found. Please run model_trainer.py first.")
        model_data = None
        return False


@app.route('/')
def index():
    if model_data is None:
        load_model()

    if model_data:
        return render_template(
            'index.html',
            genres=GENRES,
            mpaa_ratings=MPAA_RATINGS,
            major_studios=MAJOR_STUDIOS,
            known_directors=KNOWN_DIRECTORS,
            known_writers=KNOWN_WRITERS,
            ip_sources=IP_SOURCES,
            languages=LANGUAGES
        )
    else:
        return "Model not found! Please train the model first by running model_trainer.py", 500


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json or {}

        # Extract core numeric / categorical inputs
        budget = float(data.get('budget', 50000000))
        runtime = float(data.get('runtime', 120))
        genre = data.get('genre', 'Drama')
        mpaa_rating = data.get('mpaa_rating', 'PG-13')
        production_company = data.get('production_company', 'Independent Studio')
        num_production_companies = int(data.get('num_production_companies', 1))

        has_top_star = 1 if data.get('has_top_star') == 'yes' else 0
        has_ensemble_cast = 1 if data.get('has_ensemble_cast') == 'yes' else 0
        director = data.get('director', 'Other/Unknown')
        writer = data.get('writer', 'Other/Unknown')

        release_month = int(data.get('release_month', 6))
        has_competition = 1 if data.get('has_competition') == 'yes' else 0
        num_competitors = int(data.get('num_competitors', 0)) if has_competition else 0

        trailer_views = float(data.get('trailer_views', 1000000))
        expected_imdb = float(data.get('expected_imdb', 7.0))

        is_sequel = 1 if data.get('is_sequel') == 'yes' else 0
        ip_source = data.get('ip_source', 'Original')
        previous_revenue = float(data.get('previous_revenue', 0)) if is_sequel else 0

        languages_selected = data.get('languages', [])
        num_languages = len(languages_selected)

        # Determine if major studio from company name
        major_studios_list = ['Disney', 'Warner', 'Universal',
                              'Paramount', 'Fox', 'Sony', 'Columbia']
        is_major_studio = 1 if any(studio in production_company for studio in major_studios_list) else 0

        # Release season from month
        season_map = {
            1: 'Awards', 2: 'Awards', 3: 'Awards',
            4: 'Regular', 5: 'Summer', 6: 'Summer',
            7: 'Summer', 8: 'Summer', 9: 'Regular',
            10: 'Regular', 11: 'Holiday', 12: 'Holiday'
        }
        release_season = season_map.get(release_month, 'Regular')

        # Encoders from model package
        try:
            le_genre = model_data['label_encoders']['genre']
            le_season = model_data['label_encoders']['season']
        except KeyError:
            return jsonify({
                'success': False,
                'error': "Model is missing label_encoders. Delete movie_revenue_model.pkl and re-run model_trainer.py."
            }), 500

        # Encode categorical variables with safe fallback
        try:
            genre_encoded = le_genre.transform([genre])[0]
        except Exception:
            genre_encoded = 0

        try:
            season_encoded = le_season.transform([release_season])[0]
        except Exception:
            season_encoded = 0

        # Estimate votes from trailer views and expected IMDb
        votes = (trailer_views / 100.0) * (expected_imdb / 10.0)

        # Build feature dict matching training
        feature_dict = {
            'budget': budget,
            'runtime': runtime,
            'release_month': release_month,
            'imdb_score': expected_imdb,
            'num_production_companies': num_production_companies,
            'is_major_studio': is_major_studio,
            'has_top_star': has_top_star,
            'is_sequel': is_sequel,
            'votes': votes,
            'genre_encoded': genre_encoded,
            'season_encoded': season_encoded
        }

        features = model_data['features']
        input_df = pd.DataFrame([feature_dict], columns=features)

        # Base model prediction
        model = model_data['model']
        base_prediction = float(model.predict(input_df)[0])

        # Apply multipliers from high-level factors
        multiplier = 1.0

        # IP source multiplier
        ip_multipliers = {
            'Original': 1.0,
            'Book Adaptation': 1.15,
            'Comic/Graphic Novel': 1.25,
            'Remake': 1.1,
            'Video Game': 0.9,
            'TV Series': 1.2,
            'True Story': 1.05
        }
        multiplier *= ip_multipliers.get(ip_source, 1.0)

        # Ensemble cast boost
        if has_ensemble_cast:
            multiplier *= 1.1

        # Director prestige
        if director not in ['Other/Unknown']:
            multiplier *= 1.15

        # Competition penalty
        if has_competition:
            multiplier *= max(0.8, 1.0 - (num_competitors * 0.05))

        # International appeal (languages)
        if num_languages > 1:
            multiplier *= (1.0 + (num_languages * 0.05))

        # Previous film boost for sequels
        if is_sequel and previous_revenue > 0:
            sequel_boost = min(1.5, 1.0 + (previous_revenue / 500_000_000) * 0.3)
            multiplier *= sequel_boost

        final_prediction = max(0.0, base_prediction * multiplier)

        # Impact breakdown (for UI cards)
        impacts = [
            {'factor': '💰 Budget', 'impact': budget * 2.5, 'percentage': 25},
            {'factor': '🎬 Genre', 'impact': base_prediction * 0.15, 'percentage': 15},
            {
                'factor': '⭐ Star Power',
                'impact': base_prediction * 0.12 if has_top_star else 0,
                'percentage': 12 if has_top_star else 0
            },
            {
                'factor': '🎪 Studio',
                'impact': base_prediction * 0.10 if is_major_studio else 0,
                'percentage': 10 if is_major_studio else 0
            },
            {'factor': '🎯 Release Season', 'impact': base_prediction * 0.08, 'percentage': 8},
            {
                'factor': '🎬 Sequel/Franchise',
                'impact': base_prediction * 0.15 if is_sequel else 0,
                'percentage': 15 if is_sequel else 0
            },
            {
                'factor': '📚 IP Source',
                'impact': base_prediction * ((ip_multipliers.get(ip_source, 1.0) - 1.0) * 10),
                'percentage': int((ip_multipliers.get(ip_source, 1.0) - 1.0) * 100)
            },
        ]

        # Filter zero-impact and sort by magnitude
        impacts = [i for i in impacts if i['impact'] > 0]
        impacts.sort(key=lambda x: abs(x['impact']), reverse=True)

        # Chart image (base64)
        chart_url = generate_chart(impacts, final_prediction) if impacts else None

        return jsonify({
            'success': True,
            'prediction': final_prediction,
            'impacts': impacts,
            'chart': chart_url,
            'multiplier': multiplier
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


def generate_chart(impacts, final_pred):
    """Generate horizontal bar chart of impacts as base64 PNG."""
    if not impacts:
        return None

    factors = [x['factor'] for x in impacts[:8]]
    values = [x['impact'] for x in impacts[:8]]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(factors)))

    bars = ax.barh(factors, values, color=colors)
    ax.set_xlabel('Impact on Revenue ($)', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Revenue Impact Analysis\nPredicted: ${final_pred:,.0f}',
        fontsize=14,
        fontweight='bold',
        pad=20
    )

    for bar, val in zip(bars, values):
        width = bar.get_width()
        ax.text(
            width,
            bar.get_y() + bar.get_height() / 2,
            f' ${val:,.0f}',
            ha='left',
            va='center',
            fontweight='bold'
        )

    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close(fig)

    return f"data:image/png;base64,{image_base64}"


if __name__ == '__main__':
    load_model()
    app.run(debug=True, port=5000)
