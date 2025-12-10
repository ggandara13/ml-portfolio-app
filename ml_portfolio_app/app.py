"""
ML Portfolio App - Brooklyn Data Interview
==========================================
Senior Data Scientist (Retail ML) - Case Studies

Three core workstreams:
1. Predictive LTV - Customer lifetime value with limited data
2. Media Mix Model - Marketing effectiveness and budget optimization
3. Customer Segmentation - Behavioral clustering for ad targeting

Author: Gerardo Gandara
"""

import streamlit as st

st.set_page_config(
    page_title="ML Portfolio - Data Science Case Studies",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A5F;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-top: 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🔬 Data Science Portfolio</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Senior Data Scientist - Retail ML Case Studies</p>', unsafe_allow_html=True)

st.markdown("---")

# Overview
st.markdown("""
### Interview Case Studies

This portfolio demonstrates end-to-end ML capabilities across three core workstreams 
for a **major US retail client**. Each case study shows the complete data science workflow: 
data understanding, methodology selection, model building, and business translation.
""")

# Three columns for the workstreams
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    #### 🎯 Predictive LTV
    **Customer Lifetime Value**
    
    *Challenge:* Estimate LTV with only aggregate join/cancel data (no individual cohorts)
    
    *Techniques:*
    - Aggregate survival analysis
    - Churn-based LTV estimation
    - Bootstrap confidence intervals
    - Geographic segmentation
    
    *Business Value:*
    - Acquisition cost thresholds
    - Retention prioritization
    - Marketing ROI targets
    """)

with col2:
    st.markdown("""
    #### 📊 Media Mix Model
    **Marketing Effectiveness**
    
    *Challenge:* Quantify channel ROI and optimize budget allocation
    
    *Techniques:*
    - Bayesian regression (PyMC)
    - Adstock & saturation curves
    - Channel decomposition
    - Budget optimization
    
    *Business Value:*
    - Channel ROI measurement
    - Budget reallocation
    - Diminishing returns analysis
    """)

with col3:
    st.markdown("""
    #### 👥 Customer Segmentation
    **Behavioral Clustering**
    
    *Challenge:* Identify actionable segments for ad targeting
    
    *Techniques:*
    - RFM analysis
    - K-Means clustering
    - Cluster profiling
    - Segment validation
    
    *Business Value:*
    - Targeted campaigns
    - Personalized messaging
    - Lookalike audiences
    """)

st.markdown("---")

# Methodology Philosophy
st.markdown("""
### 💡 Approach Philosophy

> *"Choose the right method for the data available, while knowing exactly what you'd do with better data."*

Each case study demonstrates:

| Step | Description |
|------|-------------|
| **1. Data Understanding** | EDA, quality checks, identify limitations |
| **2. Methodology Selection** | Why this approach for this data |
| **3. Model Building** | Transparent, reproducible notebooks |
| **4. Business Translation** | Clear insights for stakeholders |
| **5. Extensions** | What's possible with more data/time |
""")

st.markdown("---")

# Technical Stack
st.markdown("### 🛠️ Technical Stack")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    **Core**
    - Python 3.10+
    - Pandas / NumPy
    - Scikit-learn
    """)

with col2:
    st.markdown("""
    **Bayesian/Stats**
    - PyMC
    - Lifelines
    - SciPy
    """)

with col3:
    st.markdown("""
    **Visualization**
    - Matplotlib
    - Plotly
    - Streamlit
    """)

with col4:
    st.markdown("""
    **MLOps**
    - Jupyter Notebooks
    - Git/GitHub
    - Reproducible pipelines
    """)

st.markdown("---")

# Sidebar
st.sidebar.markdown("""
### About This Portfolio

**Role:** Senior Data Scientist (Retail ML)  
**Client:** Major US Eyewear Retailer  
**Via:** Brooklyn Data Co

---

**Core Workstreams:**
| Module | Hours |
|--------|-------|
| pLTV | 288 |
| MMM | 288 |
| Segmentation | 216 |

---

👈 **Select a case study from the pages above**

---

*Built by Gerardo Gandara*  
*Powered by Streamlit*
""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    📧 Contact: <a href='https://github.com/ggandara13'>GitHub</a> | 
    📍 Miami, FL
</div>
""", unsafe_allow_html=True)
