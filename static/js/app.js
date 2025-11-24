document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('predictionForm');
    const resultsSection = document.getElementById('results');
    const errorBox = document.getElementById('error');
    const predictionAmount = document.getElementById('predictionAmount');
    const multiplierInfo = document.getElementById('multiplierInfo');
    const impactsList = document.getElementById('impactsList');
    const chartImage = document.getElementById('chartImage');

    const btnPredict = document.querySelector('.btn-predict');
    const loader = document.querySelector('.btn-predict .loader');

    const releaseMonth = document.getElementById('release_month');
    const seasonHint = document.getElementById('seasonHint');
    const isSequel = document.getElementById('is_sequel');
    const previousRevenueGroup = document.getElementById('previousRevenueGroup');
    const hasCompetition = document.getElementById('has_competition');
    const competitorGroup = document.getElementById('competitorGroup');

    // ---- UI helpers ----
    function setLoading(isLoading) {
        if (!btnPredict || !loader) return;
        btnPredict.disabled = isLoading;
        loader.style.display = isLoading ? 'inline-block' : 'none';
    }

    function showError(msg) {
        if (!errorBox) return;
        errorBox.textContent = msg;
        errorBox.classList.remove('hidden');
    }

    function clearError() {
        if (!errorBox) return;
        errorBox.textContent = '';
        errorBox.classList.add('hidden');
    }

    // Release month → season hint
    if (releaseMonth && seasonHint) {
        const seasonMap = {
            1: 'Awards Season',
            2: 'Awards Season',
            3: 'Awards Season',
            4: 'Regular Release',
            5: 'Summer Blockbuster',
            6: 'Summer Blockbuster',
            7: 'Summer Blockbuster',
            8: 'Summer Blockbuster',
            9: 'Regular Release',
            10: 'Regular Release',
            11: 'Holiday Season',
            12: 'Holiday Season'
        };

        function updateSeasonHint() {
            const val = parseInt(releaseMonth.value || '6', 10);
            seasonHint.textContent = `Season: ${seasonMap[val] || 'Regular Release'}`;
        }

        releaseMonth.addEventListener('change', updateSeasonHint);
        updateSeasonHint();
    }

    // Sequel toggle → previous revenue field
    if (isSequel && previousRevenueGroup) {
        function updatePreviousRevenueVisibility() {
            if (isSequel.value === 'yes') {
                previousRevenueGroup.style.display = 'block';
            } else {
                previousRevenueGroup.style.display = 'none';
            }
        }
        isSequel.addEventListener('change', updatePreviousRevenueVisibility);
        updatePreviousRevenueVisibility();
    }

    // Competition toggle → competitors field
    if (hasCompetition && competitorGroup) {
        function updateCompetitorVisibility() {
            if (hasCompetition.value === 'yes') {
                competitorGroup.style.display = 'block';
            } else {
                competitorGroup.style.display = 'none';
            }
        }
        hasCompetition.addEventListener('change', updateCompetitorVisibility);
        updateCompetitorVisibility();
    }

    // ---- Form submit ----
    if (form) {
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            clearError();
            resultsSection.classList.add('hidden');
            impactsList.innerHTML = '';
            chartImage.style.display = 'none';
            multiplierInfo.textContent = '';

            setLoading(true);

            try {
                const formData = new FormData(form);

                // Collect checkbox languages
                const languages = [];
                document.querySelectorAll('input[name="languages"]:checked')
                    .forEach(cb => languages.push(cb.value));

                const payload = {
                    budget: formData.get('budget'),
                    runtime: formData.get('runtime'),
                    genre: formData.get('genre'),
                    mpaa_rating: formData.get('mpaa_rating'),
                    production_company: formData.get('production_company'),
                    num_production_companies: formData.get('num_production_companies'),
                    has_top_star: formData.get('has_top_star'),
                    has_ensemble_cast: formData.get('has_ensemble_cast'),
                    director: formData.get('director'),
                    writer: formData.get('writer'),
                    release_month: formData.get('release_month'),
                    has_competition: formData.get('has_competition'),
                    num_competitors: formData.get('num_competitors'),
                    trailer_views: formData.get('trailer_views'),
                    expected_imdb: formData.get('expected_imdb'),
                    is_sequel: formData.get('is_sequel'),
                    ip_source: formData.get('ip_source'),
                    previous_revenue: formData.get('previous_revenue'),
                    languages: languages
                };

                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                const data = await response.json();

                if (!data.success) {
                    throw new Error(data.error || 'Prediction failed.');
                }

                // Show prediction
                const prediction = data.prediction || 0;
                const multiplier = data.multiplier || 1.0;

                predictionAmount.textContent = `$${prediction.toLocaleString(undefined, {
                    maximumFractionDigits: 0
                })}`;

                multiplierInfo.textContent = `Overall multiplier applied: ×${multiplier.toFixed(2)}`;

                // Impact cards
                impactsList.innerHTML = '';
                (data.impacts || []).forEach(item => {
                    const div = document.createElement('div');
                    div.className = 'impact-item';

                    const factor = document.createElement('div');
                    factor.className = 'impact-factor';
                    factor.textContent = item.factor;

                    const details = document.createElement('div');
                    details.className = 'impact-details';

                    const value = document.createElement('div');
                    value.className = 'impact-value';
                    value.textContent = `$${Math.round(item.impact).toLocaleString()}`;

                    const perc = document.createElement('div');
                    perc.className = 'impact-percentage';
                    perc.textContent = `${item.percentage || 0}%`;

                    details.appendChild(value);
                    details.appendChild(perc);

                    div.appendChild(factor);
                    div.appendChild(details);

                    impactsList.appendChild(div);
                });

                // Chart image
                if (data.chart) {
                    chartImage.src = data.chart;
                    chartImage.style.display = 'block';
                } else {
                    chartImage.style.display = 'none';
                }

                resultsSection.classList.remove('hidden');
                resultsSection.scrollIntoView({ behavior: 'smooth' });
            } catch (err) {
                console.error(err);
                showError(err.message || 'Something went wrong while predicting.');
            } finally {
                setLoading(false);
            }
        });
    }
});

// Global reset function used by the button onclick in HTML
function resetForm() {
    const form = document.getElementById('predictionForm');
    const resultsSection = document.getElementById('results');
    const errorBox = document.getElementById('error');
    const impactsList = document.getElementById('impactsList');
    const chartImage = document.getElementById('chartImage');
    const multiplierInfo = document.getElementById('multiplierInfo');
    const seasonHint = document.getElementById('seasonHint');
    const competitorGroup = document.getElementById('competitorGroup');
    const previousRevenueGroup = document.getElementById('previousRevenueGroup');
    const releaseMonth = document.getElementById('release_month');
    const hasCompetition = document.getElementById('has_competition');
    const isSequel = document.getElementById('is_sequel');

    if (form) form.reset();
    if (resultsSection) resultsSection.classList.add('hidden');
    if (errorBox) {
        errorBox.textContent = '';
        errorBox.classList.add('hidden');
    }
    if (impactsList) impactsList.innerHTML = '';
    if (chartImage) {
        chartImage.src = '';
        chartImage.style.display = 'none';
    }
    if (multiplierInfo) multiplierInfo.textContent = '';

    if (seasonHint && releaseMonth) {
        releaseMonth.value = '6';
        seasonHint.textContent = 'Season: Summer Blockbuster';
    }
    if (competitorGroup && hasCompetition) {
        hasCompetition.value = 'no';
        competitorGroup.style.display = 'none';
    }
    if (previousRevenueGroup && isSequel) {
        isSequel.value = 'no';
        previousRevenueGroup.style.display = 'none';
    }

    window.scrollTo({ top: 0, behavior: 'smooth' });
}
