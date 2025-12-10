# 🔬 ML Portfolio - Data Science Case Studies

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ml-portfolio-ggandara.streamlit.app)

Interactive portfolio demonstrating end-to-end ML capabilities across three core workstreams for retail analytics.

## 📊 Case Studies

### 1. 🎯 Predictive LTV (Customer Lifetime Value)
Estimate customer lifetime value using only aggregate join/cancel data — no individual cohort records needed.

**Key Techniques:**
- Aggregate survival analysis
- Churn-based LTV estimation
- Bootstrap confidence intervals
- Geographic segmentation

**Business Value:**
- Acquisition cost thresholds
- Retention prioritization
- Marketing ROI targets

---

### 2. 📊 Media Mix Model (MMM)
Quantify marketing channel effectiveness and optimize budget allocation.

**Key Techniques:**
- Bayesian regression (PyMC)
- Adstock & saturation curves
- Channel decomposition
- Budget optimization

**Business Value:**
- Channel ROI measurement
- Budget reallocation recommendations
- Diminishing returns analysis

---

### 3. 👥 Customer Segmentation
Behavioral clustering to identify actionable customer segments for targeted marketing.

**Key Techniques:**
- RFM analysis
- K-Means clustering
- PCA visualization
- Segment profiling

**Business Value:**
- Targeted campaigns
- Personalized messaging
- Lookalike audience creation

---

## 🚀 Run Locally

```bash
# Clone the repository
git clone https://github.com/ggandara13/ml-portfolio-app.git
cd ml-portfolio-app

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Streamlit** - Interactive web app
- **Pandas / NumPy** - Data manipulation
- **Plotly** - Interactive visualizations
- **Scikit-learn** - ML algorithms
- **SciPy** - Statistical analysis

---

## 📁 Project Structure

```
ml_portfolio_app/
├── app.py                          # Main landing page
├── pages/
│   ├── 1_🎯_LTV_Analysis.py        # Predictive LTV module
│   ├── 2_📊_Media_Mix_Model.py     # MMM module
│   └── 3_👥_Customer_Segmentation.py # Segmentation module
├── requirements.txt
└── README.md
```

---

## 📋 Methodology Highlights

### LTV with Limited Data
When individual cohort data isn't available, we can still estimate LTV:
1. Estimate membership base using steady-state assumption
2. Calculate monthly churn rate from aggregate flows
3. Apply exponential survival assumption
4. Bootstrap for confidence intervals

### Bayesian MMM
Advantages over frequentist approach:
- Full posterior distributions (uncertainty quantification)
- Prior knowledge incorporation
- Better small-data performance
- Interpretable results

### Customer Segmentation
Beyond basic RFM:
- Include behavioral features (channel preference, product mix)
- Use silhouette score for optimal k selection
- Validate with business stakeholders
- Create actionable segment profiles

---

## 👤 Author

**Gerardo Gandara**  
Senior Data Scientist | Miami, FL  
[GitHub](https://github.com/ggandara13)

---

## 📄 License

MIT License - feel free to use and adapt for your own portfolio!
