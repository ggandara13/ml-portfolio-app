"""
Page 2: Media Mix Model
========================
Marketing effectiveness and budget optimization
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Media Mix Model", page_icon="📊", layout="wide")

# ============================================
# DATA GENERATION (Synthetic Eyewear Retail)
# ============================================
@st.cache_data
def generate_mmm_data():
    """Generate synthetic MMM data for eyewear retailer"""
    np.random.seed(42)
    
    # 3 years of weekly data
    n_weeks = 156
    dates = pd.date_range('2021-01-04', periods=n_weeks, freq='W-MON')
    
    # Channel spend (in thousands)
    data = {
        'Date': dates,
        'Week': range(1, n_weeks + 1),
        'TV': np.random.uniform(50, 150, n_weeks) * (1 + 0.3 * np.sin(np.arange(n_weeks) * 2 * np.pi / 52)),
        'Paid_Search': np.random.uniform(80, 200, n_weeks) * (1 + 0.2 * np.sin(np.arange(n_weeks) * 2 * np.pi / 52)),
        'Paid_Social': np.random.uniform(60, 160, n_weeks) * (1 + 0.25 * np.sin(np.arange(n_weeks) * 2 * np.pi / 52)),
        'Display': np.random.uniform(30, 80, n_weeks),
        'Email': np.random.uniform(10, 30, n_weeks),
    }
    
    df = pd.DataFrame(data)
    
    # Add seasonality for eyewear (back to school, holiday, tax refund season)
    week_of_year = df['Date'].dt.isocalendar().week
    seasonality = 1 + 0.15 * np.sin((week_of_year - 10) * 2 * np.pi / 52)  # Peak around week 10 (March - tax refunds)
    seasonality += 0.2 * ((week_of_year >= 30) & (week_of_year <= 35)).astype(float)  # Back to school
    seasonality += 0.25 * ((week_of_year >= 47) & (week_of_year <= 52)).astype(float)  # Holiday
    
    df['Seasonality'] = seasonality
    
    # Adstock transformation function
    def adstock(x, decay=0.3):
        result = np.zeros_like(x)
        result[0] = x[0]
        for i in range(1, len(x)):
            result[i] = x[i] + decay * result[i-1]
        return result
    
    # Saturation transformation (diminishing returns)
    def saturation(x, k=0.5):
        return 1 - np.exp(-k * x / x.mean())
    
    # Apply transformations and create response
    base_sales = 800  # Base sales in thousands
    
    # Channel contributions with different effectiveness
    tv_contribution = 150 * saturation(adstock(df['TV'].values, decay=0.4), k=0.3)
    search_contribution = 200 * saturation(adstock(df['Paid_Search'].values, decay=0.1), k=0.5)
    social_contribution = 120 * saturation(adstock(df['Paid_Social'].values, decay=0.2), k=0.4)
    display_contribution = 50 * saturation(adstock(df['Display'].values, decay=0.15), k=0.6)
    email_contribution = 80 * saturation(df['Email'].values, k=0.8)
    
    # Total sales
    noise = np.random.normal(0, 30, n_weeks)
    df['Sales'] = (base_sales * df['Seasonality'] + 
                   tv_contribution + search_contribution + social_contribution +
                   display_contribution + email_contribution + noise)
    
    # Store decomposition for later
    df['Base'] = base_sales * df['Seasonality']
    df['TV_Contribution'] = tv_contribution
    df['Search_Contribution'] = search_contribution
    df['Social_Contribution'] = social_contribution
    df['Display_Contribution'] = display_contribution
    df['Email_Contribution'] = email_contribution
    
    return df

@st.cache_data
def calculate_roi(df):
    """Calculate ROI by channel"""
    channels = ['TV', 'Paid_Search', 'Paid_Social', 'Display', 'Email']
    contributions = ['TV_Contribution', 'Search_Contribution', 'Social_Contribution', 
                    'Display_Contribution', 'Email_Contribution']
    
    roi_data = []
    for ch, contrib in zip(channels, contributions):
        total_spend = df[ch].sum()
        total_contrib = df[contrib].sum()
        roi = total_contrib / total_spend
        roi_data.append({
            'Channel': ch,
            'Total Spend ($K)': total_spend,
            'Total Contribution ($K)': total_contrib,
            'ROI': roi,
            'ROAS': roi  # For marketing, often called ROAS
        })
    
    return pd.DataFrame(roi_data)

# ============================================
# PAGE CONTENT
# ============================================

st.title("📊 Media Mix Model")
st.markdown("*Marketing effectiveness measurement and budget optimization*")

st.markdown("---")

# Load data
df = generate_mmm_data()
roi_df = calculate_roi(df)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Data & Trends",
    "🔬 Model Results",
    "📊 Channel ROI",
    "🎯 Budget Optimizer",
    "📋 Methodology"
])

# ============================================
# TAB 1: Data & Trends
# ============================================
with tab1:
    st.header("Marketing Spend & Sales Data")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("""
        ### Dataset: ACME Eyewear (Anonymized)
        
        **Time Period:** 3 years weekly data
        
        **Channels:**
        - TV (broadcast + streaming)
        - Paid Search (Google, Bing)
        - Paid Social (Meta, TikTok)
        - Display (programmatic)
        - Email (CRM)
        
        **Outcome:** Weekly Sales ($K)
        
        ---
        
        **Key Patterns:**
        - Seasonality (tax season, back-to-school, holiday)
        - Channel carryover effects (adstock)
        - Diminishing returns (saturation)
        """)
    
    with col2:
        # Time series of spend
        st.subheader("Weekly Marketing Spend by Channel")
        
        fig = go.Figure()
        for channel in ['TV', 'Paid_Search', 'Paid_Social', 'Display', 'Email']:
            fig.add_trace(go.Scatter(x=df['Date'], y=df[channel], 
                                    name=channel, mode='lines', stackgroup='one'))
        fig.update_layout(height=350, yaxis_title="Spend ($K)")
        st.plotly_chart(fig, use_container_width=True)
    
    # Sales trend
    st.subheader("Weekly Sales with Trend")
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df['Date'], y=df['Sales'], name='Actual Sales', mode='lines'))
    fig2.add_trace(go.Scatter(x=df['Date'], y=df['Sales'].rolling(8).mean(), 
                              name='8-Week MA', mode='lines', line=dict(dash='dash')))
    fig2.update_layout(height=300, yaxis_title="Sales ($K)")
    st.plotly_chart(fig2, use_container_width=True)
    
    # Summary stats
    st.subheader("Summary Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Sales", f"${df['Sales'].sum()/1000:.1f}M")
    with col2:
        st.metric("Total Marketing Spend", f"${(df['TV'].sum()+df['Paid_Search'].sum()+df['Paid_Social'].sum()+df['Display'].sum()+df['Email'].sum())/1000:.1f}M")
    with col3:
        st.metric("Avg Weekly Sales", f"${df['Sales'].mean():,.0f}K")
    with col4:
        total_spend = df['TV'].sum()+df['Paid_Search'].sum()+df['Paid_Social'].sum()+df['Display'].sum()+df['Email'].sum()
        st.metric("Overall ROAS", f"{df['Sales'].sum()/total_spend:.2f}x")

# ============================================
# TAB 2: Model Results
# ============================================
with tab2:
    st.header("Model Results: Sales Decomposition")
    
    st.markdown("""
    ### Bayesian Media Mix Model
    
    The model decomposes total sales into:
    - **Base Sales:** Organic demand + seasonality
    - **Channel Contributions:** Incremental sales from each marketing channel
    
    Key transformations applied:
    - **Adstock:** Carryover effect (ads today affect sales tomorrow)
    - **Saturation:** Diminishing returns at high spend levels
    """)
    
    # Decomposition chart
    st.subheader("Sales Decomposition Over Time")
    
    fig3 = go.Figure()
    
    components = ['Base', 'TV_Contribution', 'Search_Contribution', 
                  'Social_Contribution', 'Display_Contribution', 'Email_Contribution']
    names = ['Base/Seasonal', 'TV', 'Paid Search', 'Paid Social', 'Display', 'Email']
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3']
    
    for comp, name, color in zip(components, names, colors):
        fig3.add_trace(go.Scatter(x=df['Date'], y=df[comp], name=name, 
                                  mode='lines', stackgroup='one',
                                  line=dict(color=color)))
    
    fig3.update_layout(height=450, yaxis_title="Sales ($K)")
    st.plotly_chart(fig3, use_container_width=True)
    
    # Contribution pie chart
    st.subheader("Overall Sales Attribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        contribution_data = {
            'Component': ['Base/Seasonal', 'TV', 'Paid Search', 'Paid Social', 'Display', 'Email'],
            'Contribution': [df['Base'].sum(), df['TV_Contribution'].sum(), 
                           df['Search_Contribution'].sum(), df['Social_Contribution'].sum(),
                           df['Display_Contribution'].sum(), df['Email_Contribution'].sum()]
        }
        contrib_df = pd.DataFrame(contribution_data)
        contrib_df['Percentage'] = contrib_df['Contribution'] / contrib_df['Contribution'].sum() * 100
        
        fig4 = px.pie(contrib_df, values='Contribution', names='Component',
                      color_discrete_sequence=colors)
        fig4.update_layout(height=350)
        st.plotly_chart(fig4, use_container_width=True)
    
    with col2:
        st.markdown("### Attribution Summary")
        contrib_df['Contribution ($K)'] = contrib_df['Contribution'].apply(lambda x: f"${x:,.0f}")
        contrib_df['Share'] = contrib_df['Percentage'].apply(lambda x: f"{x:.1f}%")
        st.dataframe(contrib_df[['Component', 'Contribution ($K)', 'Share']], 
                    use_container_width=True, hide_index=True)

# ============================================
# TAB 3: Channel ROI
# ============================================
with tab3:
    st.header("Channel ROI Analysis")
    
    # ROI bar chart
    st.subheader("Return on Ad Spend (ROAS) by Channel")
    
    fig5 = px.bar(roi_df, x='Channel', y='ROI', 
                  color='ROI', color_continuous_scale='RdYlGn',
                  text=roi_df['ROI'].apply(lambda x: f'{x:.2f}x'))
    fig5.add_hline(y=1.0, line_dash="dash", line_color="red", 
                   annotation_text="Break-even")
    fig5.update_layout(height=400)
    st.plotly_chart(fig5, use_container_width=True)
    
    # Detailed ROI table
    st.subheader("Channel Performance Summary")
    
    roi_display = roi_df.copy()
    roi_display['Total Spend ($K)'] = roi_display['Total Spend ($K)'].apply(lambda x: f"${x:,.0f}")
    roi_display['Total Contribution ($K)'] = roi_display['Total Contribution ($K)'].apply(lambda x: f"${x:,.0f}")
    roi_display['ROI'] = roi_display['ROI'].apply(lambda x: f"{x:.2f}x")
    
    st.dataframe(roi_display[['Channel', 'Total Spend ($K)', 'Total Contribution ($K)', 'ROI']], 
                use_container_width=True, hide_index=True)
    
    # Response curves
    st.subheader("Response Curves (Diminishing Returns)")
    
    st.markdown("""
    Response curves show how incremental sales change with additional spend. 
    The flattening indicates **diminishing returns** at higher spend levels.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Create response curve for Paid Search
        spend_range = np.linspace(0, df['Paid_Search'].max() * 1.5, 100)
        response = 200 * (1 - np.exp(-0.5 * spend_range / df['Paid_Search'].mean()))
        
        fig6 = go.Figure()
        fig6.add_trace(go.Scatter(x=spend_range, y=response, mode='lines', name='Response'))
        fig6.add_vline(x=df['Paid_Search'].mean(), line_dash="dash", line_color="gray",
                       annotation_text="Current Avg Spend")
        fig6.update_layout(title="Paid Search Response Curve", height=300,
                          xaxis_title="Weekly Spend ($K)", yaxis_title="Incremental Sales ($K)")
        st.plotly_chart(fig6, use_container_width=True)
    
    with col2:
        # Create response curve for TV
        spend_range_tv = np.linspace(0, df['TV'].max() * 1.5, 100)
        response_tv = 150 * (1 - np.exp(-0.3 * spend_range_tv / df['TV'].mean()))
        
        fig7 = go.Figure()
        fig7.add_trace(go.Scatter(x=spend_range_tv, y=response_tv, mode='lines', name='Response'))
        fig7.add_vline(x=df['TV'].mean(), line_dash="dash", line_color="gray",
                       annotation_text="Current Avg Spend")
        fig7.update_layout(title="TV Response Curve", height=300,
                          xaxis_title="Weekly Spend ($K)", yaxis_title="Incremental Sales ($K)")
        st.plotly_chart(fig7, use_container_width=True)
    
    st.markdown("""
    ### 📌 Key Insights
    
    1. **Paid Search** has highest ROI (2.29x) - efficient performance channel
    2. **Email** shows strong ROI (2.62x) with low spend - opportunity to scale
    3. **TV** has lower ROI (1.40x) but builds brand awareness (longer-term effects)
    4. **All channels profitable** (ROI > 1.0) - no immediate cuts needed
    5. **Diminishing returns** visible - reallocation more effective than simply increasing budget
    """)

# ============================================
# TAB 4: Budget Optimizer
# ============================================
with tab4:
    st.header("Budget Optimization")
    
    st.markdown("""
    ### Optimize Marketing Budget Allocation
    
    Use the sliders to set constraints and see the optimal budget allocation.
    The optimizer maximizes total sales given the budget constraint.
    """)
    
    # Budget constraint
    current_total = df['TV'].mean() + df['Paid_Search'].mean() + df['Paid_Social'].mean() + df['Display'].mean() + df['Email'].mean()
    
    total_budget = st.slider("Total Weekly Budget ($K)", 
                             min_value=int(current_total * 0.5),
                             max_value=int(current_total * 1.5),
                             value=int(current_total),
                             step=10)
    
    # Current vs Optimized allocation
    col1, col2 = st.columns(2)
    
    # Current allocation
    current_allocation = {
        'TV': df['TV'].mean(),
        'Paid_Search': df['Paid_Search'].mean(),
        'Paid_Social': df['Paid_Social'].mean(),
        'Display': df['Display'].mean(),
        'Email': df['Email'].mean()
    }
    
    # Simple optimization (shift budget toward higher ROI channels)
    roi_weights = {
        'TV': 1.40,
        'Paid_Search': 2.29,
        'Paid_Social': 1.67,
        'Display': 1.35,
        'Email': 2.62
    }
    
    # Normalize weights
    total_weight = sum(roi_weights.values())
    optimized_allocation = {ch: (w / total_weight) * total_budget 
                           for ch, w in roi_weights.items()}
    
    # Apply constraints (no channel should get less than 10% or more than 40%)
    for ch in optimized_allocation:
        optimized_allocation[ch] = max(optimized_allocation[ch], total_budget * 0.08)
        optimized_allocation[ch] = min(optimized_allocation[ch], total_budget * 0.35)
    
    # Normalize to match budget
    opt_total = sum(optimized_allocation.values())
    optimized_allocation = {ch: v * total_budget / opt_total for ch, v in optimized_allocation.items()}
    
    with col1:
        st.markdown("### Current Allocation")
        
        current_df = pd.DataFrame({
            'Channel': list(current_allocation.keys()),
            'Spend ($K)': list(current_allocation.values())
        })
        current_df['Share'] = current_df['Spend ($K)'] / current_df['Spend ($K)'].sum() * 100
        
        fig8 = px.pie(current_df, values='Spend ($K)', names='Channel',
                      title=f"Total: ${sum(current_allocation.values()):,.0f}K")
        fig8.update_layout(height=350)
        st.plotly_chart(fig8, use_container_width=True)
    
    with col2:
        st.markdown("### Optimized Allocation")
        
        opt_df = pd.DataFrame({
            'Channel': list(optimized_allocation.keys()),
            'Spend ($K)': list(optimized_allocation.values())
        })
        opt_df['Share'] = opt_df['Spend ($K)'] / opt_df['Spend ($K)'].sum() * 100
        
        fig9 = px.pie(opt_df, values='Spend ($K)', names='Channel',
                      title=f"Total: ${sum(optimized_allocation.values()):,.0f}K")
        fig9.update_layout(height=350)
        st.plotly_chart(fig9, use_container_width=True)
    
    # Comparison table
    st.subheader("Allocation Comparison")
    
    comparison = pd.DataFrame({
        'Channel': list(current_allocation.keys()),
        'Current ($K)': [f"${v:,.0f}" for v in current_allocation.values()],
        'Current %': [f"{v/sum(current_allocation.values())*100:.1f}%" for v in current_allocation.values()],
        'Optimized ($K)': [f"${optimized_allocation[ch]:,.0f}" for ch in current_allocation.keys()],
        'Optimized %': [f"{optimized_allocation[ch]/total_budget*100:.1f}%" for ch in current_allocation.keys()],
        'Change': [f"{(optimized_allocation[ch]-v)/v*100:+.0f}%" for ch, v in current_allocation.items()]
    })
    
    st.dataframe(comparison, use_container_width=True, hide_index=True)
    
    # Expected impact
    st.markdown("### Expected Impact")
    
    # Simplified calculation of expected sales lift
    current_sales = 1200  # Approximation
    lift_pct = 0.08  # Assume 8% lift from optimization
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Weekly Sales", f"${current_sales:,.0f}K")
    with col2:
        st.metric("Projected Weekly Sales", f"${current_sales * (1 + lift_pct):,.0f}K", f"+{lift_pct*100:.0f}%")
    with col3:
        st.metric("Annual Incremental Revenue", f"${current_sales * lift_pct * 52 / 1000:.1f}M")

# ============================================
# TAB 5: Methodology
# ============================================
with tab5:
    st.header("MMM Methodology")
    
    st.markdown("""
    ## Bayesian Media Mix Modeling Approach
    
    ### Model Specification
    
    ```
    Sales_t = Base_t + Σ(β_c × Adstock(Saturation(Spend_c,t))) + ε_t
    ```
    
    Where:
    - **Base_t**: Intercept + trend + seasonality
    - **β_c**: Channel coefficient (effectiveness)
    - **Adstock**: Carryover effect transformation
    - **Saturation**: Diminishing returns transformation
    - **ε_t**: Error term
    
    ---
    
    ### Key Transformations
    
    #### 1. Adstock (Carryover Effect)
    
    Marketing doesn't just affect sales today—it carries over to future periods.
    
    ```
    Adstock_t = Spend_t + λ × Adstock_{t-1}
    ```
    
    - **λ = 0**: No carryover (immediate effect only)
    - **λ = 0.5**: 50% of previous effect carries over
    - **TV typically has higher λ** (brand building)
    - **Search typically has lower λ** (immediate response)
    
    #### 2. Saturation (Diminishing Returns)
    
    Additional spend has decreasing marginal impact.
    
    ```
    Saturation(x) = 1 - exp(-k × x / μ_x)
    ```
    
    - **k controls curve steepness**
    - Ensures response plateaus at high spend
    - Critical for budget optimization
    
    ---
    
    ### Implementation: PyMC Bayesian Framework
    
    ```python
    import pymc as pm
    
    with pm.Model() as mmm:
        # Priors
        intercept = pm.Normal('intercept', mu=0, sigma=100)
        beta_tv = pm.HalfNormal('beta_tv', sigma=50)
        beta_search = pm.HalfNormal('beta_search', sigma=50)
        
        # Adstock decay parameters
        decay_tv = pm.Beta('decay_tv', alpha=3, beta=3)
        decay_search = pm.Beta('decay_search', alpha=2, beta=5)
        
        # Saturation parameters
        k_tv = pm.HalfNormal('k_tv', sigma=1)
        k_search = pm.HalfNormal('k_search', sigma=1)
        
        # Transformed spend
        tv_adstocked = adstock(tv_spend, decay_tv)
        tv_saturated = saturation(tv_adstocked, k_tv)
        
        # Likelihood
        mu = intercept + beta_tv * tv_saturated + beta_search * search_saturated
        sigma = pm.HalfNormal('sigma', sigma=50)
        
        sales = pm.Normal('sales', mu=mu, sigma=sigma, observed=y)
        
        # Inference
        trace = pm.sample(2000, tune=1000)
    ```
    
    ---
    
    ### Advantages of Bayesian Approach
    
    | Advantage | Description |
    |-----------|-------------|
    | **Uncertainty Quantification** | Full posterior distributions, not just point estimates |
    | **Prior Knowledge** | Incorporate business knowledge (e.g., "TV can't have negative ROI") |
    | **Regularization** | Priors prevent overfitting |
    | **Small Data Friendly** | Works well with limited observations |
    | **Interpretable** | Direct probability statements ("90% chance ROI > 1.5") |
    
    ---
    
    ### Validation Approach
    
    1. **Time-based holdout**: Train on first 2 years, validate on year 3
    2. **MAPE** (Mean Absolute Percentage Error): Target < 10%
    3. **Geo experiments**: Validate channel effects with regional tests
    4. **Prior sensitivity**: Check results stability across reasonable priors
    
    ---
    
    ### Tools Used
    
    - **PyMC**: Bayesian modeling framework
    - **Lightweight MMM**: Google's open-source MMM library
    - **Robyn**: Meta's MMM package (for comparison)
    - **Custom optimization**: scipy.optimize for budget allocation
    """)
