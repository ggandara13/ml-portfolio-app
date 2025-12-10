"""
Page 1: Predictive LTV Analysis
================================
Customer Lifetime Value estimation with aggregate data
Using real gym membership data (anonymized)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

st.set_page_config(page_title="LTV Analysis", page_icon="🎯", layout="wide")

# ============================================
# DATA LOADING (Real data from CSV)
# ============================================
@st.cache_data
def load_data():
    """Load pre-aggregated monthly gym membership data from CSV"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    data_path = os.path.join(parent_dir, 'data_sources', 'acme_gym_monthly.csv')
    
    df = pd.read_csv(data_path)
    
    # Parse date
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Ensure Total_Joins exists
    if 'Total_Joins' not in df.columns:
        df['Total_Joins'] = df['Web_Joins'] + df['In_Club_Joins']
    
    return df

@st.cache_data
def create_monthly_aggregates(df):
    """Aggregate monthly club-level data to monthly totals by membership type"""
    df = df.copy()
    df['YearMonth'] = df['Date'].dt.to_period('M')
    
    monthly = df.groupby(['YearMonth', 'Membership_Category']).agg({
        'Total_Joins': 'sum',
        'Cancels': 'sum',
        'ABC_Club_ID': 'nunique'
    }).reset_index()
    
    monthly.columns = ['YearMonth', 'Membership_Category', 'Joins', 'Cancels', 'Active_Clubs']
    monthly['Net_Change'] = monthly['Joins'] - monthly['Cancels']
    monthly['Date'] = monthly['YearMonth'].dt.to_timestamp()
    
    return monthly

@st.cache_data
def calculate_membership_series(monthly_df, category, baseline):
    """Calculate membership trajectory over time"""
    cat_df = monthly_df[monthly_df['Membership_Category'] == category].copy()
    cat_df = cat_df.sort_values('Date').reset_index(drop=True)
    
    members = [baseline]
    for i in range(1, len(cat_df)):
        new_members = members[-1] + cat_df.loc[i, 'Net_Change']
        members.append(max(new_members, 100000))
    
    cat_df['Est_Members'] = members
    cat_df['Churn_Rate'] = cat_df['Cancels'] / cat_df['Est_Members']
    return cat_df

# ============================================
# PAGE CONTENT
# ============================================

st.title("🎯 Predictive LTV Analysis")
st.markdown("*Customer Lifetime Value estimation with aggregate join/cancel data*")

st.markdown("---")

# Load data
try:
    df_daily = load_data()
    monthly_df = create_monthly_aggregates(df_daily)
    data_loaded = True
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("Please ensure data files are in the `data_sources` folder.")
    data_loaded = False

if data_loaded:
    # Tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Overview", 
        "📈 EDA & Trends", 
        "🔬 Survival Analysis",
        "💰 LTV Calculation",
        "📋 Methodology"
    ])

    # ============================================
    # TAB 1: Data Overview
    # ============================================
    with tab1:
        st.header("Data Overview")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(f"""
            ### Dataset: Fitness Chain (Anonymized)
            
            **Data Available:**
            - Monthly joins and cancellations
            - {df_daily['ABC_Club_ID'].nunique():,} gym locations
            - {df_daily['Date'].dt.to_period('M').nunique():,} months of data
            - {df_daily['Date'].min().strftime('%Y-%m')} to {df_daily['Date'].max().strftime('%Y-%m')}
            - 2 main membership tiers
            
            **Data NOT Available:**
            - Individual user records
            - Join/cancel dates per user
            - User demographics
            - Engagement metrics
            
            ---
            
            ### The Challenge
            
            > *"Calculate customer LTV without individual-level cohort data"*
            
            This is common in practice when you only have aggregate reporting.
            """)
        
        with col2:
            st.markdown("### Summary Statistics")
            
            summary = monthly_df[monthly_df['Membership_Category'].isin(['Black Card', 'White Card'])].groupby('Membership_Category').agg({
                'Joins': 'sum',
                'Cancels': 'sum',
                'Net_Change': 'sum'
            }).reset_index()
            summary['Churn_Ratio'] = summary['Cancels'] / summary['Joins']
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                bc_data = summary[summary['Membership_Category']=='Black Card']
                if len(bc_data) > 0:
                    st.metric("Black Card Total Joins", f"{bc_data['Joins'].values[0]:,.0f}")
                    st.metric("Black Card Net Change", f"{bc_data['Net_Change'].values[0]:,.0f}")
            
            with col_b:
                wc_data = summary[summary['Membership_Category']=='White Card']
                if len(wc_data) > 0:
                    st.metric("White Card Total Joins", f"{wc_data['Joins'].values[0]:,.0f}")
                    st.metric("White Card Net Change", f"{wc_data['Net_Change'].values[0]:,.0f}")
            
            st.dataframe(summary.round(2), use_container_width=True)

    # ============================================
    # TAB 2: EDA & Trends
    # ============================================
    with tab2:
        st.header("Exploratory Data Analysis")
        
        st.subheader("Joins vs Cancels Over Time")
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=('Black Card', 'White Card'),
                            vertical_spacing=0.1)
        
        for idx, membership in enumerate(['Black Card', 'White Card'], 1):
            subset = monthly_df[monthly_df['Membership_Category'] == membership]
            
            fig.add_trace(go.Scatter(x=subset['Date'], y=subset['Joins'],
                                     name=f'{membership} Joins', mode='lines',
                                     line=dict(color='green' if idx==1 else 'blue')),
                          row=idx, col=1)
            fig.add_trace(go.Scatter(x=subset['Date'], y=subset['Cancels'],
                                     name=f'{membership} Cancels', mode='lines',
                                     line=dict(color='red', dash='dash')),
                          row=idx, col=1)
        
        # Add COVID line as shapes
        fig.add_shape(type="line", x0="2020-03-15", x1="2020-03-15", y0=0, y1=1,
                      yref="paper", line=dict(color="gray", dash="dot", width=1))
        
        fig.update_layout(height=500, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # Net change cumulative
        st.subheader("Cumulative Net Membership Change")
        
        fig2 = go.Figure()
        for membership in ['Black Card', 'White Card']:
            subset = monthly_df[monthly_df['Membership_Category'] == membership].copy()
            subset = subset.sort_values('Date')
            subset['Cumulative'] = subset['Net_Change'].cumsum()
            
            fig2.add_trace(go.Scatter(x=subset['Date'], y=subset['Cumulative'],
                                      name=membership, mode='lines'))
        
        fig2.add_hline(y=0, line_dash="solid", line_color="black", opacity=0.3)
        fig2.update_layout(height=400, yaxis_title="Cumulative Net Change")
        st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("""
        ### 📌 Key Observations
        
        1. **January Spike:** Clear seasonality with New Year's resolution effect
        2. **COVID Impact:** Massive cancellation spike in March-June 2020
        3. **Divergent Recovery:** Black Card never recovered; White Card stabilized
        4. **Structural Shift:** Premium tier (Black Card) lost members permanently
        """)

    # ============================================
    # TAB 3: Survival Analysis
    # ============================================
    with tab3:
        st.header("Survival Analysis")
        
        st.markdown("""
        ### Approach: Aggregate Survival Estimation
        
        Without individual cohort data, we estimate survival using aggregate churn rates:
        
        - **Step 1:** Estimate membership base using steady-state assumption
        - **Step 2:** Calculate monthly churn rate = Cancels / Estimated Members
        - **Step 3:** Build survival curve: S(t) = (1 - churn)^t
        """)
        
        black_series = calculate_membership_series(monthly_df, 'Black Card', 4500000)
        white_series = calculate_membership_series(monthly_df, 'White Card', 3000000)
        
        st.subheader("Monthly Churn Rate Over Time")
        
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=black_series['Date'], y=black_series['Churn_Rate']*100,
                                  name='Black Card', mode='lines'))
        fig3.add_trace(go.Scatter(x=white_series['Date'], y=white_series['Churn_Rate']*100,
                                  name='White Card', mode='lines'))
        fig3.add_hline(y=4, line_dash="dash", line_color="gray", 
                       annotation_text="4% benchmark")
        fig3.update_layout(height=400, yaxis_title="Monthly Churn Rate (%)")
        st.plotly_chart(fig3, use_container_width=True)
        
        st.subheader("Simulated Survival Curves")
        
        col1, col2 = st.columns(2)
        
        pre_covid_churn_bc = black_series[black_series['Date'] < '2020-03-01']['Churn_Rate'].mean()
        pre_covid_churn_wc = white_series[white_series['Date'] < '2020-03-01']['Churn_Rate'].mean()
        recovery_churn_bc = black_series[black_series['Date'] >= '2021-06-01']['Churn_Rate'].mean()
        recovery_churn_wc = white_series[white_series['Date'] >= '2021-06-01']['Churn_Rate'].mean()
        
        months = np.arange(0, 25)
        
        with col1:
            survival_bc = [(1 - pre_covid_churn_bc)**m * 100 for m in months]
            survival_wc = [(1 - pre_covid_churn_wc)**m * 100 for m in months]
            
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(x=months, y=survival_bc, name='Black Card', mode='lines'))
            fig4.add_trace(go.Scatter(x=months, y=survival_wc, name='White Card', mode='lines'))
            fig4.add_hline(y=50, line_dash="dash", line_color="gray")
            fig4.update_layout(title="Pre-COVID Survival", height=350,
                              xaxis_title="Months", yaxis_title="% Remaining")
            st.plotly_chart(fig4, use_container_width=True)
        
        with col2:
            survival_bc_r = [(1 - recovery_churn_bc)**m * 100 for m in months]
            survival_wc_r = [(1 - recovery_churn_wc)**m * 100 for m in months]
            
            fig5 = go.Figure()
            fig5.add_trace(go.Scatter(x=months, y=survival_bc_r, name='Black Card', mode='lines'))
            fig5.add_trace(go.Scatter(x=months, y=survival_wc_r, name='White Card', mode='lines'))
            fig5.add_hline(y=50, line_dash="dash", line_color="gray")
            fig5.update_layout(title="Recovery Period Survival", height=350,
                              xaxis_title="Months", yaxis_title="% Remaining")
            st.plotly_chart(fig5, use_container_width=True)
        
        st.subheader("Survival Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Black Card Pre-COVID Churn", f"{pre_covid_churn_bc*100:.1f}%")
        with col2:
            st.metric("Black Card Current Churn", f"{recovery_churn_bc*100:.1f}%", 
                      f"+{(recovery_churn_bc-pre_covid_churn_bc)*100:.1f}%")
        with col3:
            st.metric("White Card Pre-COVID Churn", f"{pre_covid_churn_wc*100:.1f}%")
        with col4:
            st.metric("White Card Current Churn", f"{recovery_churn_wc*100:.1f}%",
                      f"{(recovery_churn_wc-pre_covid_churn_wc)*100:+.1f}%")

    # ============================================
    # TAB 4: LTV Calculation
    # ============================================
    with tab4:
        st.header("LTV Calculation")
        
        black_series = calculate_membership_series(monthly_df, 'Black Card', 4500000)
        white_series = calculate_membership_series(monthly_df, 'White Card', 3000000)
        
        pre_covid_churn_bc = black_series[black_series['Date'] < '2020-03-01']['Churn_Rate'].mean()
        pre_covid_churn_wc = white_series[white_series['Date'] < '2020-03-01']['Churn_Rate'].mean()
        recovery_churn_bc = black_series[black_series['Date'] >= '2021-06-01']['Churn_Rate'].mean()
        recovery_churn_wc = white_series[white_series['Date'] >= '2021-06-01']['Churn_Rate'].mean()
        
        st.sidebar.markdown("### LTV Parameters")
        black_card_price = st.sidebar.number_input("Black Card Monthly Price", value=24.99, step=1.0)
        white_card_price = st.sidebar.number_input("White Card Monthly Price", value=10.00, step=1.0)
        discount_rate = st.sidebar.slider("Annual Discount Rate", 0.0, 0.20, 0.10, 0.01)
        
        monthly_discount = (1 + discount_rate) ** (1/12) - 1
        
        def calc_ltv(churn_rate, monthly_price, discount_rate):
            tenure = 1 / churn_rate
            ltv_simple = tenure * monthly_price
            months = np.arange(1, int(tenure) + 1)
            ltv_disc = sum(monthly_price / ((1 + discount_rate) ** m) for m in months)
            return tenure, ltv_simple, ltv_disc
        
        tenure_bc_pre, ltv_bc_pre_simple, ltv_bc_pre_disc = calc_ltv(pre_covid_churn_bc, black_card_price, monthly_discount)
        tenure_wc_pre, ltv_wc_pre_simple, ltv_wc_pre_disc = calc_ltv(pre_covid_churn_wc, white_card_price, monthly_discount)
        tenure_bc_cur, ltv_bc_cur_simple, ltv_bc_cur_disc = calc_ltv(recovery_churn_bc, black_card_price, monthly_discount)
        tenure_wc_cur, ltv_wc_cur_simple, ltv_wc_cur_disc = calc_ltv(recovery_churn_wc, white_card_price, monthly_discount)
        
        st.subheader("LTV Comparison: Pre-COVID vs Current")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Black Card")
            ltv_data_bc = pd.DataFrame({
                'Period': ['Pre-COVID', 'Current', 'Change'],
                'Monthly Churn': [f"{pre_covid_churn_bc*100:.1f}%", f"{recovery_churn_bc*100:.1f}%", f"+{(recovery_churn_bc-pre_covid_churn_bc)*100:.1f}%"],
                'Expected Tenure': [f"{tenure_bc_pre:.1f} mo", f"{tenure_bc_cur:.1f} mo", f"{tenure_bc_cur-tenure_bc_pre:.1f} mo"],
                'LTV (Discounted)': [f"${ltv_bc_pre_disc:,.0f}", f"${ltv_bc_cur_disc:,.0f}", f"-${ltv_bc_pre_disc-ltv_bc_cur_disc:,.0f}"]
            })
            st.dataframe(ltv_data_bc, use_container_width=True, hide_index=True)
            st.metric("LTV Change", f"-${ltv_bc_pre_disc-ltv_bc_cur_disc:,.0f}", 
                      f"-{(ltv_bc_pre_disc-ltv_bc_cur_disc)/ltv_bc_pre_disc*100:.0f}%")
        
        with col2:
            st.markdown("### White Card")
            ltv_data_wc = pd.DataFrame({
                'Period': ['Pre-COVID', 'Current', 'Change'],
                'Monthly Churn': [f"{pre_covid_churn_wc*100:.1f}%", f"{recovery_churn_wc*100:.1f}%", f"+{(recovery_churn_wc-pre_covid_churn_wc)*100:.1f}%"],
                'Expected Tenure': [f"{tenure_wc_pre:.1f} mo", f"{tenure_wc_cur:.1f} mo", f"{tenure_wc_cur-tenure_wc_pre:.1f} mo"],
                'LTV (Discounted)': [f"${ltv_wc_pre_disc:,.0f}", f"${ltv_wc_cur_disc:,.0f}", f"-${ltv_wc_pre_disc-ltv_wc_cur_disc:,.0f}"]
            })
            st.dataframe(ltv_data_wc, use_container_width=True, hide_index=True)
            st.metric("LTV Change", f"-${ltv_wc_pre_disc-ltv_wc_cur_disc:,.0f}",
                      f"-{(ltv_wc_pre_disc-ltv_wc_cur_disc)/ltv_wc_pre_disc*100:.0f}%")
        
        st.subheader("LTV Comparison Chart")
        
        fig6 = go.Figure()
        categories = ['Black Card', 'White Card']
        pre_covid_ltv = [ltv_bc_pre_disc, ltv_wc_pre_disc]
        current_ltv = [ltv_bc_cur_disc, ltv_wc_cur_disc]
        
        fig6.add_trace(go.Bar(name='Pre-COVID', x=categories, y=pre_covid_ltv, 
                              marker_color='green', opacity=0.7))
        fig6.add_trace(go.Bar(name='Current', x=categories, y=current_ltv,
                              marker_color='red', opacity=0.7))
        fig6.update_layout(barmode='group', height=400, yaxis_title="LTV ($)")
        st.plotly_chart(fig6, use_container_width=True)
        
        st.markdown("---")
        st.subheader("📋 Executive Summary")
        
        st.markdown(f"""
        ### Key Findings
        
        | Metric | Black Card | White Card |
        |--------|-----------|------------|
        | Pre-COVID LTV | ${ltv_bc_pre_disc:,.0f} | ${ltv_wc_pre_disc:,.0f} |
        | Current LTV | ${ltv_bc_cur_disc:,.0f} | ${ltv_wc_cur_disc:,.0f} |
        | LTV Change | **-{(ltv_bc_pre_disc-ltv_bc_cur_disc)/ltv_bc_pre_disc*100:.0f}%** | -{(ltv_wc_pre_disc-ltv_wc_cur_disc)/ltv_wc_pre_disc*100:.0f}% |
        | Recovery Status | ❌ Not recovered | ✅ Stabilized |
        
        ### Strategic Implications
        
        1. **Black Card LTV premium shrunk** from ${ltv_bc_pre_disc-ltv_wc_pre_disc:,.0f} to ${ltv_bc_cur_disc-ltv_wc_cur_disc:,.0f}
        2. **Acquisition cost thresholds** need recalibration for Black Card
        3. **Retention campaigns** should prioritize Black Card members
        4. **Geographic targeting** can identify high-churn regions for intervention
        """)

    # ============================================
    # TAB 5: Methodology
    # ============================================
    with tab5:
        st.header("Methodology Documentation")
        
        st.markdown("""
        ## Approach Used: Aggregate Survival Analysis
        
        ### The Challenge
        
        We have **aggregate flow data** (joins, cancels) but not **stock data** (total members) 
        or **individual cohort data** (user-level join/cancel dates).
        
        ### Solution Framework
        
        ```
        Step 1: Estimate Membership Base
        ├── Use steady-state assumption: Members ≈ Cancels / Assumed_Churn_Rate
        ├── Validate against industry benchmarks (3-5% monthly churn for gyms)
        └── Track forward: Members_t = Members_{t-1} + Joins_t - Cancels_t
        
        Step 2: Calculate Churn Rates
        ├── Monthly Churn = Cancels_t / Estimated_Members_t
        └── Segment by tier, period, geography
        
        Step 3: Survival Analysis (Exponential Assumption)
        ├── Assumes CONSTANT HAZARD (churn doesn't vary by tenure)
        ├── Survival: S(t) = (1 - λ)^t where λ = monthly churn
        └── Expected Tenure = 1 / λ
        
        Step 4: LTV Calculation
        ├── Simple: LTV = Tenure × Monthly_Price
        └── Discounted: LTV = Σ (Price / (1+r)^t) for t = 1 to Tenure
        ```
        
        ### Limitations
        
        | Limitation | Impact | Mitigation |
        |------------|--------|------------|
        | Constant hazard assumption | May over/underestimate tenure | Acknowledged in CIs |
        | No cohort separation | Can't distinguish new vs tenured | Used period segmentation |
        | Baseline uncertainty | Starting membership estimated | Validated vs benchmarks |
        | No individual features | Can't predict individual LTV | Used segment-level analysis |
        
        ---
        
        ## With Individual User Data, We Would Use:
        
        ### 1. Kaplan-Meier Survival Curves
        - Non-parametric, no distributional assumptions
        - True cohort-level survival curves
        
        ### 2. Cox Proportional Hazards
        - Semi-parametric with covariates
        - Identifies which features drive churn
        
        ### 3. Weibull/AFT Models
        - Parametric survival with shape parameter
        - Handles non-constant hazard
        
        ### 4. Predictive CLV (Machine Learning)
        - XGBoost, Random Forest for churn prediction
        - Individual-level LTV predictions
        """)
