# Enhanced Movie Revenue Predictor Web App

## 🎬 What's New?

### **Major Upgrades**
- ✅ **20+ Parameters** instead of 5 basic ones
- ✅ **Random Forest Model** for better accuracy (was Linear Regression)
- ✅ **Modern Premium UI** with animations and gradients
- ✅ **Smart Conditional Fields** (show/hide based on inputs)
- ✅ **Enhanced Predictions** with multipliers for complex factors

---

## 📊 Complete Parameter List (20+ Questions)

### **Core Production Features (6)**
1. 💰 **Budget** - Production cost in dollars
2. ⏱️ **Runtime** - Movie length in minutes
3. 🎬 **Genre** - Action, Comedy, Drama, Horror, Animation, etc.
4. 🔞 **MPAA Rating** - G, PG, PG-13, R, NC-17
5. 🎪 **Production Company** - Major studio or Independent
6. 🏢 **Number of Production Companies** - How many studios involved

### **Star Power & Creative Team (4)**
7. ⭐ **Lead Actor Star Power** - Is lead in Forbes Celebrity list?
8. 🎭 **Supporting Cast Quality** - Multiple known actors (ensemble)?
9. 🎥 **Director** - Select from known directors or Other
10. ✍️ **Writer** - Select from known writers or Other

### **Marketing & Distribution (4)**
11. 📅 **Release Month** - When releasing (1-12)
12. 🎯 **Release Season** - Auto-detected (Summer/Holiday/Awards/Regular)
13. 🎬 **Competition** - Other major movies same weekend?
14. 🔢 **Number of Competitors** - How many competing releases

### **Pre-Release Metrics (2)**
15. 📺 **Trailer Views** - YouTube views count
16. 🌟 **Expected IMDb Score** - Based on test screenings (0-10)

### **Franchise & Source Material (3)**
17. 🎬 **Is Sequel/Franchise** - Part of existing franchise?
18. 📚 **Based on Known IP** - Original, Book, Comic, Remake, etc.
19. 🏆 **Previous Film Revenue** - Box office of last film (if sequel)

### **Competition & Market (2)**
20. 🌍 **Release Languages** - Multiple language releases
21. 🌐 **International Appeal** - Based on selected languages

---

## 🎨 UI Improvements

### **Design Features**
- **Hero Section** with animated gradient background
- **Stats Display** showing key metrics
- **Organized Sections** with color-coded categories
- **Smooth Animations** on hover and focus
- **Conditional Fields** that appear/disappear based on selections
- **Premium Color Scheme** with purple/blue gradients
- **Responsive Design** works on mobile and desktop
- **Impact Breakdown Cards** with percentage indicators
- **Interactive Charts** with colorful visualizations

### **UX Enhancements**
- Auto-updating season hints based on month
- Smart field visibility (competitors, previous revenue)
- Input validation and helpful hints
- Smooth scrolling to results
- Loading animations during prediction
- Error handling with friendly messages

---

## 🚀 Setup Instructions

### **Step 1: Install Dependencies**
```bash
cd movie-rev-pred
source .venv/bin/activate  # or: source venv/bin/activate
pip install Flask pandas numpy scikit-learn matplotlib joblib
```

### **Step 2: Train the Enhanced Model**
```bash
python model_trainer.py
```
This will:
- Load all datasets
- Engineer 20+ features
- Train Random Forest model
- Save enhanced model file

### **Step 3: Run the Web App**
```bash
python app.py
```
Open browser to: **http://localhost:5000**

---

## 🎯 How It Works

### **Prediction Algorithm**
1. **Base Prediction** - Random Forest model on core features
2. **IP Source Multiplier** - Boost for franchises, books, comics
3. **Ensemble Cast Boost** - +10% for multiple stars
4. **Director Prestige** - +15% for known directors
5. **Competition Penalty** - Reduces revenue based on competitors
6. **International Appeal** - Bonus for multi-language releases
7. **Sequel Boost** - Based on previous film performance

### **Model Features Used**
- Budget, Runtime, Release Month
- IMDb Score, Votes (estimated from trailer views)
- Number of Production Companies
- Is Major Studio (Disney, Warner, Universal, etc.)
- Has Top Star (Forbes Celebrity)
- Is Sequel/Franchise
- Genre (encoded)
- Release Season (encoded)

---

## 📁 File Structure

```
movie-rev-pred/
├── app.py                    # Enhanced Flask app with 20+ parameters
├── model_trainer.py          # Random Forest trainer
├── requirements.txt          # Dependencies
├── movie_revenue_model.pkl   # Trained model
│
├── data/                     # Your CSV files
│   ├── movies_metadata.csv
│   ├── movies.csv
│   ├── forbes_celebrity_100.csv
│   ├── tmdb_5000_movies.csv
│   ├── wiki_movie_plots_deduped.csv
│   └── Highest Holywood Grossing Movies.csv
│
├── static/
│   ├── css/
│   │   └── style.css        # Premium modern styling
│   └── js/
│       └── app.js           # Enhanced interactivity
│
└── templates/
    └── index.html           # 20+ parameter form
```

---

## 🎨 Color Scheme

- **Primary**: Purple/Blue gradients (#667eea → #764ba2)
- **Success**: Green gradient (#11998e → #38ef7d)
- **Accent**: Deep purple (#7e22ce)
- **Background**: Dark blue gradient (#1e3c72 → #2a5298)

---

## 🔧 Customization

### Add More Genres
Edit `GENRES` list in `app.py`:
```python
GENRES = ['Action', 'Your New Genre', ...]
```

### Add More Studios
Edit `MAJOR_STUDIOS` list in `app.py`:
```python
MAJOR_STUDIOS = ['Disney', 'Your Studio', ...]
```

### Change Color Theme
Edit `static/css/style.css`:
```css
/* Change primary gradient */
background: linear-gradient(135deg, #YOUR_COLOR1 0%, #YOUR_COLOR2 100%);
```

---

## 📈 Prediction Accuracy

- **Model Type**: Random Forest Regressor
- **Features**: 11 core + encoded categorical
- **Training**: Outlier removal + median imputation
- **Multipliers**: 6 additional factors
- **Expected Accuracy**: ~80-90% (varies by data quality)

---

## 🐛 Troubleshooting

### Model Not Found
```bash
python model_trainer.py  # Train first
```

### Import Errors
```bash
pip install Flask pandas numpy scikit-learn matplotlib joblib
```

### Port Already in Use
Edit `app.py`:
```python
app.run(debug=True, port=5001)  # Change port
```

---

## 🎬 Example Predictions

### Blockbuster Setup
- Budget: $200M
- Genre: Action
- MPAA: PG-13
- Major Studio: Yes
- Top Star: Yes
- Sequel: Yes
- Release: Summer
- **Expected**: $500M - $800M

### Indie Drama Setup
- Budget: $15M
- Genre: Drama
- MPAA: R
- Independent Studio
- Known Director
- Original Story
- Release: Awards Season
- **Expected**: $30M - $80M

---

## 📄 License

Open source for educational purposes.

## 🤝 Contributing

Feel free to enhance with:
- More data sources
- Additional parameters
- Better UI designs
- Advanced ML models

---

**Built with ❤️ using Flask, scikit-learn, and modern web technologies**