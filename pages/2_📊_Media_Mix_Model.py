"""
Page 2: Media Mix Model
=======================
Marketing effectiveness analysis using Robyn (Meta) with PyMC validation
Real retail client data (anonymized)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

st.set_page_config(page_title="Media Mix Model", page_icon="📊", layout="wide")

# ============================================
# DATA LOADING (Real data from CSV)
# ============================================
@st.cache_data
def load_mmm_data():
    """Load real MMM data from CSV"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    data_path = os.path.join(parent_dir, 'data_sources', 'MMM_Input_sample.csv')
    
    df = pd.read_csv(data_path)
    df['week_start'] = pd.to_datetime(df['week_start'])
    
    return df

# ============================================
# ROBYN MODEL RESULTS (Model 5_164_1)
# ============================================
ROBYN_RESULTS = {
    'model_id': '5_164_1',
    'adj_r2': 0.9122,
    'nrmse': 0.0505,
    'decomp_rssd': 0.1062,
    'total_revenue': 7_639_007_846,
    'baseline_pct': 93.5,
    'media_pct': 6.5,
    
    # Channel results (from Robyn one-pager)
    'channels': {
        'Print_Alternate': {'spend': 53_444_752, 'effect_share': 0.314, 'spend_share': 0.290, 'roi': 3.36, 'contribution': 164_000_000},
        'Print_Preferred': {'spend': 51_808_663, 'effect_share': 0.263, 'spend_share': 0.281, 'roi': 3.12, 'contribution': 148_000_000},
        'OA_MAILER_S': {'spend': 25_620_918, 'effect_share': 0.162, 'spend_share': 0.139, 'roi': 3.53, 'contribution': 83_800_000},
        'Social_S': {'spend': 20_670_721, 'effect_share': 0.115, 'spend_share': 0.112, 'roi': 2.85, 'contribution': 54_200_000},
        'OLV_S': {'spend': 5_669_762, 'effect_share': 0.101, 'spend_share': 0.031, 'roi': 9.66, 'contribution': 50_100_000},
        'Search_S': {'spend': 8_930_441, 'effect_share': 0.048, 'spend_share': 0.048, 'roi': 1.71, 'contribution': 14_000_000},
        'LinearTV_S': {'spend': 4_811_473, 'effect_share': 0.027, 'spend_share': 0.026, 'roi': 2.27, 'contribution': 10_000_000},
        'CTV_New_S': {'spend': 1_585_908, 'effect_share': 0.019, 'spend_share': 0.009, 'roi': 9.50, 'contribution': 13_800_000},
        'Display_and_Programmatic_S': {'spend': 4_424_787, 'effect_share': 0.005, 'spend_share': 0.024, 'roi': 0.00, 'contribution': 0},
        'Cardlytics_Amex_S': {'spend': 7_524_493, 'effect_share': 0.001, 'spend_share': 0.041, 'roi': 0.00, 'contribution': 0},
    },
    
    # Adstock parameters (from Robyn)
    'adstock': {
        'Print_Alternate': {'carryover': 0.24, 'immediate': 0.76},
        'Print_Preferred': {'carryover': 0.34, 'immediate': 0.66},
        'OA_MAILER_S': {'carryover': 0.02, 'immediate': 0.98},
        'Social_S': {'carryover': 0.01, 'immediate': 0.99},
        'Search_S': {'carryover': 0.03, 'immediate': 0.97},
        'OLV_S': {'carryover': 0.17, 'immediate': 0.83},
        'LinearTV_S': {'carryover': 0.26, 'immediate': 0.74},
        'CTV_New_S': {'carryover': 0.05, 'immediate': 0.95},
        'Display_and_Programmatic_S': {'carryover': 0.00, 'immediate': 1.00},
        'Cardlytics_Amex_S': {'carryover': 0.31, 'immediate': 0.69},
    }
}

# PyMC Validation confidence levels
PYMC_VALIDATION = {
    'Print_Alternate': {'pymc_roi': 4.96, 'validated': True, 'confidence': 'HIGH'},
    'Print_Preferred': {'pymc_roi': 4.56, 'validated': True, 'confidence': 'HIGH'},
    'OA_MAILER_S': {'pymc_roi': 3.55, 'validated': True, 'confidence': 'HIGH'},
    'Social_S': {'pymc_roi': 5.74, 'validated': True, 'confidence': 'MEDIUM'},
    'OLV_S': {'pymc_roi': 10.73, 'validated': True, 'confidence': 'MEDIUM'},
    'Search_S': {'pymc_roi': 4.31, 'validated': False, 'confidence': 'LOW'},
    'LinearTV_S': {'pymc_roi': 2.80, 'validated': True, 'confidence': 'HIGH'},
    'CTV_New_S': {'pymc_roi': 4.66, 'validated': True, 'confidence': 'MEDIUM'},
    'Display_and_Programmatic_S': {'pymc_roi': 0.24, 'validated': False, 'confidence': 'LOW'},
    'Cardlytics_Amex_S': {'pymc_roi': 0.18, 'validated': False, 'confidence': 'LOW'},
}

# ============================================
# PAGE CONTENT
# ============================================

st.title("📊 Media Mix Model")
st.markdown("*Marketing effectiveness analysis - Robyn (Meta) with PyMC validation*")

st.markdown("---")

# Load data
try:
    df = load_mmm_data()
    data_loaded = True
except Exception as e:
    st.error(f"Error loading data: {e}")
    data_loaded = False

if data_loaded:
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Overview",
        "📈 EDA & Patterns",
        "🎯 Model Results",
        "💡 Business Insights",
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
            ### Dataset: ACME Retail (Anonymized)
            
            **Time Period:**
            - {df['week_start'].min().strftime('%Y-%m-%d')} to {df['week_start'].max().strftime('%Y-%m-%d')}
            - {len(df)} weeks of data
            
            **Media Channels:** 10
            - Print (Preferred & Alternate)
            - Direct Mail (OA Mailer)
            - Digital (Social, Search, OLV, Display)
            - TV (Linear & CTV)
            - Cardlytics/Amex
            
            **Outcome:** Weekly Revenue
            
            **Context:** Promo Days (store events)
            """)
            
            total_spend = sum([ch['spend'] for ch in ROBYN_RESULTS['channels'].values()])
            st.metric("Total Media Spend", f"${total_spend:,.0f}")
            st.metric("Analysis Period", f"{len(df)} weeks")
        
        with col2:
            st.subheader("Spend by Channel")
            
            spend_data = pd.DataFrame([
                {'Channel': ch, 'Spend': data['spend']}
                for ch, data in ROBYN_RESULTS['channels'].items()
            ]).sort_values('Spend', ascending=True)
            
            fig = px.bar(spend_data, x='Spend', y='Channel', orientation='h',
                        color='Spend', color_continuous_scale='Blues')
            fig.update_layout(height=400, showlegend=False)
            fig.update_xaxes(tickformat='$,.0f')
            st.plotly_chart(fig, use_container_width=True)
    
    # ============================================
    # TAB 2: EDA & Patterns
    # ============================================
    with tab2:
        st.header("Exploratory Data Analysis")
        
        # Revenue over time
        st.subheader("Revenue Over Time")
        
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=df['week_start'], y=df['revenue']/1e6,
                                  mode='lines', name='Revenue',
                                  line=dict(color='#636EFA')))
        fig1.update_layout(height=350, yaxis_title="Revenue ($M)",
                          xaxis_title="")
        st.plotly_chart(fig1, use_container_width=True)
        
        # Media spend patterns
        st.subheader("Media Spend Patterns Over Time")
        
        media_cols = ['Print_Preferred', 'Print_Alternate', 'OA_MAILER_S', 
                     'Social_S', 'Search_S', 'OLV_S', 'LinearTV_S', 'CTV_New_S']
        
        # Stacked area chart
        fig2 = go.Figure()
        colors = px.colors.qualitative.Set2
        
        for i, col in enumerate(media_cols):
            if col in df.columns:
                fig2.add_trace(go.Scatter(
                    x=df['week_start'], y=df[col]/1e6,
                    mode='lines', name=col.replace('_S', '').replace('_', ' '),
                    stackgroup='one',
                    line=dict(color=colors[i % len(colors)])
                ))
        
        fig2.update_layout(height=400, yaxis_title="Spend ($M)")
        st.plotly_chart(fig2, use_container_width=True)
        
        # Correlations
        st.subheader("Channel Correlations with Revenue")
        
        col1, col2 = st.columns(2)
        
        with col1:
            corr_data = []
            for col in media_cols:
                if col in df.columns:
                    corr = df[col].corr(df['revenue'])
                    corr_data.append({'Channel': col.replace('_S', ''), 'Correlation': corr})
            
            corr_df = pd.DataFrame(corr_data).sort_values('Correlation', ascending=True)
            
            fig3 = px.bar(corr_df, x='Correlation', y='Channel', orientation='h',
                         color='Correlation', color_continuous_scale='RdYlGn',
                         range_color=[-0.5, 0.8])
            fig3.update_layout(height=350, title="Raw Correlation with Revenue")
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            # Spend distribution
            spend_dist = []
            for col in media_cols:
                if col in df.columns:
                    non_zero = (df[col] > 0).sum() / len(df) * 100
                    spend_dist.append({'Channel': col.replace('_S', ''), 'Active Weeks %': non_zero})
            
            dist_df = pd.DataFrame(spend_dist).sort_values('Active Weeks %', ascending=True)
            
            fig4 = px.bar(dist_df, x='Active Weeks %', y='Channel', orientation='h',
                         color='Active Weeks %', color_continuous_scale='Greens')
            fig4.update_layout(height=350, title="Channel Activity (% weeks with spend)")
            st.plotly_chart(fig4, use_container_width=True)
        
        # Key EDA insights
        st.markdown("""
        ### 📌 Key EDA Findings
        
        1. **Sparse Channels:** Print and LinearTV have low activity (<40% of weeks)
        2. **Consistent Digital:** Social, Search run consistently (>90% of weeks)
        3. **CTV Growth:** CTV spend started mid-2023, growing channel
        4. **Multicollinearity Risk:** Display & Cardlytics have 99% temporal overlap with Social
        """)
    
    # ============================================
    # TAB 3: Model Results
    # ============================================
    with tab3:
        st.header("Robyn Model Results")
        
        st.markdown(f"""
        ### Model Performance: **{ROBYN_RESULTS['model_id']}**
        
        | Metric | Value |
        |--------|-------|
        | Adjusted R² | **{ROBYN_RESULTS['adj_r2']:.2%}** |
        | NRMSE | {ROBYN_RESULTS['nrmse']:.2%} |
        | DECOMP.RSSD | {ROBYN_RESULTS['decomp_rssd']:.4f} |
        """)
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Model R²", f"{ROBYN_RESULTS['adj_r2']:.1%}")
        with col2:
            st.metric("Baseline %", f"{ROBYN_RESULTS['baseline_pct']:.1f}%")
        with col3:
            st.metric("Media Contribution", f"{ROBYN_RESULTS['media_pct']:.1f}%")
        with col4:
            total_media_revenue = sum([ch['contribution'] for ch in ROBYN_RESULTS['channels'].values()])
            st.metric("Media Revenue", f"${total_media_revenue/1e6:.0f}M")
        
        st.markdown("---")
        
        # ROI Chart
        st.subheader("Channel ROI with PyMC Validation")
        
        roi_data = []
        for ch, data in ROBYN_RESULTS['channels'].items():
            validation = PYMC_VALIDATION.get(ch, {})
            roi_data.append({
                'Channel': ch.replace('_S', '').replace('_', ' '),
                'Robyn ROI': data['roi'],
                'PyMC ROI': validation.get('pymc_roi', 0),
                'Confidence': validation.get('confidence', 'N/A'),
                'Spend': data['spend'],
                'Contribution': data['contribution']
            })
        
        roi_df = pd.DataFrame(roi_data).sort_values('Robyn ROI', ascending=True)
        
        # Grouped bar chart
        fig5 = go.Figure()
        
        fig5.add_trace(go.Bar(
            y=roi_df['Channel'],
            x=roi_df['Robyn ROI'],
            name='Robyn',
            orientation='h',
            marker_color='#636EFA'
        ))
        
        fig5.add_trace(go.Bar(
            y=roi_df['Channel'],
            x=roi_df['PyMC ROI'],
            name='PyMC',
            orientation='h',
            marker_color='#EF553B',
            opacity=0.7
        ))
        
        fig5.update_layout(
            height=450,
            barmode='group',
            xaxis_title='ROI (Revenue per $1 Spend)',
            title='ROI Comparison: Robyn vs PyMC Validation'
        )
        st.plotly_chart(fig5, use_container_width=True)
        
        # Spend vs Effect Share
        st.subheader("Spend Share vs Effect Share")
        
        share_data = pd.DataFrame([
            {
                'Channel': ch.replace('_S', '').replace('_', ' '),
                'Spend Share': data['spend_share'] * 100,
                'Effect Share': data['effect_share'] * 100,
                'ROI': data['roi']
            }
            for ch, data in ROBYN_RESULTS['channels'].items()
        ])
        
        fig6 = go.Figure()
        
        fig6.add_trace(go.Bar(
            name='Spend Share',
            y=share_data['Channel'],
            x=share_data['Spend Share'],
            orientation='h',
            marker_color='#636EFA'
        ))
        
        fig6.add_trace(go.Bar(
            name='Effect Share',
            y=share_data['Channel'],
            x=share_data['Effect Share'],
            orientation='h',
            marker_color='#00CC96'
        ))
        
        # Add ROI as text
        for i, row in share_data.iterrows():
            fig6.add_annotation(
                x=max(row['Spend Share'], row['Effect Share']) + 2,
                y=row['Channel'],
                text=f"ROI: {row['ROI']:.1f}x",
                showarrow=False,
                font=dict(size=10)
            )
        
        fig6.update_layout(
            height=450,
            barmode='group',
            xaxis_title='Share (%)',
            title='Efficiency Analysis: Spend vs Effect'
        )
        st.plotly_chart(fig6, use_container_width=True)
        
        # Adstock / Carryover
        st.subheader("Immediate vs Carryover Response")
        
        adstock_df = pd.DataFrame([
            {
                'Channel': ch.replace('_S', '').replace('_', ' '),
                'Immediate': data['immediate'] * 100,
                'Carryover': data['carryover'] * 100
            }
            for ch, data in ROBYN_RESULTS['adstock'].items()
        ])
        
        fig7 = go.Figure()
        
        fig7.add_trace(go.Bar(
            name='Carryover',
            y=adstock_df['Channel'],
            x=adstock_df['Carryover'],
            orientation='h',
            marker_color='#FFA15A'
        ))
        
        fig7.add_trace(go.Bar(
            name='Immediate',
            y=adstock_df['Channel'],
            x=adstock_df['Immediate'],
            orientation='h',
            marker_color='#636EFA'
        ))
        
        fig7.update_layout(
            height=400,
            barmode='stack',
            xaxis_title='Response %',
            title='Adstock: How Long Does Media Effect Last?'
        )
        st.plotly_chart(fig7, use_container_width=True)
    
    # ============================================
    # TAB 4: Business Insights
    # ============================================
    with tab4:
        st.header("Business Insights & Recommendations")
        
        st.markdown("""
        ### 📊 ROI Summary by Confidence Level
        
        Results validated using independent PyMC Bayesian model.
        """)
        
        # High confidence
        st.markdown("#### ✅ HIGH CONFIDENCE (PyMC Validated)")
        
        high_conf = [ch for ch, v in PYMC_VALIDATION.items() if v['confidence'] == 'HIGH']
        high_data = []
        for ch in high_conf:
            robyn_roi = ROBYN_RESULTS['channels'][ch]['roi']
            pymc_roi = PYMC_VALIDATION[ch]['pymc_roi']
            spend = ROBYN_RESULTS['channels'][ch]['spend']
            high_data.append({
                'Channel': ch.replace('_S', '').replace('_', ' '),
                'Robyn ROI': f"{robyn_roi:.2f}x",
                'PyMC ROI': f"{pymc_roi:.2f}x",
                'Spend': f"${spend:,.0f}",
                'Recommendation': '✅ Trust for planning'
            })
        
        st.dataframe(pd.DataFrame(high_data), use_container_width=True, hide_index=True)
        
        # Medium confidence
        st.markdown("#### ⚠️ MEDIUM CONFIDENCE (Direction Confirmed)")
        
        med_conf = [ch for ch, v in PYMC_VALIDATION.items() if v['confidence'] == 'MEDIUM']
        med_data = []
        for ch in med_conf:
            robyn_roi = ROBYN_RESULTS['channels'][ch]['roi']
            pymc_roi = PYMC_VALIDATION[ch]['pymc_roi']
            spend = ROBYN_RESULTS['channels'][ch]['spend']
            med_data.append({
                'Channel': ch.replace('_S', '').replace('_', ' '),
                'Robyn ROI': f"{robyn_roi:.2f}x",
                'PyMC ROI': f"{pymc_roi:.2f}x",
                'Spend': f"${spend:,.0f}",
                'Recommendation': '⚠️ Use conservative estimate'
            })
        
        st.dataframe(pd.DataFrame(med_data), use_container_width=True, hide_index=True)
        
        # Low confidence
        st.markdown("#### ❓ LOW CONFIDENCE (Needs Investigation)")
        
        low_conf = [ch for ch, v in PYMC_VALIDATION.items() if v['confidence'] == 'LOW']
        low_data = []
        for ch in low_conf:
            robyn_roi = ROBYN_RESULTS['channels'][ch]['roi']
            pymc_roi = PYMC_VALIDATION[ch]['pymc_roi']
            spend = ROBYN_RESULTS['channels'][ch]['spend']
            low_data.append({
                'Channel': ch.replace('_S', '').replace('_', ' '),
                'Robyn ROI': f"{robyn_roi:.2f}x",
                'PyMC ROI': f"{pymc_roi:.2f}x",
                'Spend': f"${spend:,.0f}",
                'Recommendation': '❓ Consider geo-test'
            })
        
        st.dataframe(pd.DataFrame(low_data), use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Key findings
        st.subheader("🎯 Key Strategic Findings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **CTV vs Linear TV:**
            - CTV ROI: **9.5x** vs Linear TV: **2.27x**
            - CTV has only 5% carryover (immediate response)
            - Linear TV has 26% carryover (brand building)
            - **Recommendation:** Shift budget to CTV for performance
            
            **Print Channels:**
            - Both Print types show solid 3.1-3.4x ROI
            - High carryover (24-34%) indicates brand effect
            - **Recommendation:** Maintain current spend
            """)
        
        with col2:
            st.markdown("""
            **Digital Challenges:**
            - Display & Cardlytics show 0x ROI in Robyn
            - Likely absorbed by Social due to 99% overlap
            - **Recommendation:** Run geo-holdout test
            
            **OLV Performance:**
            - Highest ROI at 9.66x
            - Only 3% of total spend
            - **Recommendation:** Test incremental spend
            """)
        
        st.markdown("---")
        
        # Executive summary
        st.subheader("📋 Executive Summary")
        
        st.markdown(f"""
        ### Media Mix Model Results
        
        **Model Quality:** R² = {ROBYN_RESULTS['adj_r2']:.1%} (Excellent fit)
        
        **Total Media Spend:** ${sum([ch['spend'] for ch in ROBYN_RESULTS['channels'].values()]):,.0f}
        
        **Total Media-Driven Revenue:** ${sum([ch['contribution'] for ch in ROBYN_RESULTS['channels'].values()]):,.0f}
        
        **Overall Media ROI:** {sum([ch['contribution'] for ch in ROBYN_RESULTS['channels'].values()]) / sum([ch['spend'] for ch in ROBYN_RESULTS['channels'].values()]):.2f}x
        
        ---
        
        ### Top Recommendations
        
        1. **Increase CTV allocation** - 9.5x ROI with room to grow
        2. **Test OLV scale** - Currently only 3% of spend but highest ROI
        3. **Maintain Print** - Reliable 3x+ ROI, validated by PyMC
        4. **Geo-test Display/Cardlytics** - Can't separate effect from Social
        5. **Investigate Search** - Large discrepancy between Robyn (1.7x) and PyMC (4.3x)
        """)
    
    # ============================================
    # TAB 5: Methodology
    # ============================================
    with tab5:
        st.header("Methodology")
        
        st.markdown("""
        ## Modeling Approach
        
        ### Primary Model: Robyn (Meta Open Source)
        
        **Why Robyn?**
        - Ridge regression with Nevergrad optimization
        - Built-in adstock (Weibull PDF) and saturation (Hill function)
        - Multi-objective optimization (fit vs decomposition stability)
        - Automated hyperparameter tuning (6,000 iterations × 5 trials)
        
        **Model Specification:**
        ```
        Revenue_t = Baseline_t + Σ(β_c × Hill(Adstock(Spend_c,t))) + Controls_t + ε_t
        ```
        
        **Adstock (Weibull PDF):**
        - Captures delayed response and carryover
        - Shape parameter controls decay rate
        - Scale parameter controls peak timing
        
        **Saturation (Hill Function):**
        - Captures diminishing returns
        - Alpha controls steepness
        - Gamma controls inflection point
        
        ---
        
        ### Validation Model: PyMC (Bayesian)
        
        **Purpose:** Independent validation of channel effects
        
        **Approach:**
        - Same data, different methodology
        - Calibrated priors based on Robyn targets
        - Full posterior distributions for uncertainty
        
        **Key Insight:** PyMC with loose priors tends to overfit small-spend channels.
        This validates that Robyn's regularization is appropriate, not artificial.
        
        ---
        
        ### Confidence Framework
        
        | Level | Criteria | Action |
        |-------|----------|--------|
        | HIGH | PyMC within 50% of Robyn | Use for planning |
        | MEDIUM | Same direction, magnitude differs | Use conservative estimate |
        | LOW | Large discrepancy or absorbed | Run geo-test before acting |
        
        ---
        
        ### Known Limitations
        
        1. **Multicollinearity:** Display/Cardlytics run same weeks as Social
           - Model can't separate effects
           - Attributes to strongest correlated channel (Social)
        
        2. **Sparse Channels:** Print, LinearTV have <40% activity weeks
           - Less statistical power
           - Wider confidence intervals
        
        3. **CTV Coverage:** Only 45% of analysis window has CTV spend
           - ROI estimate has higher uncertainty
           - Direction (CTV > LinearTV) is confident
        
        ---
        
        ### Recommended Next Steps
        
        1. **Geo-holdout tests** for Display, Cardlytics, Search
        2. **Incrementality test** for OLV (highest ROI, low spend)
        3. **Budget optimizer** scenario planning with Robyn
        4. **Refresh model** quarterly as more CTV data accumulates
        """)
