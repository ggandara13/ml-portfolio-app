"""
Page 3: Store Segmentation & Performance Analysis
=================================================
Advanced ML analysis using real retail store data:
- XGBoost performance prediction with SHAP explainability
- K-Means clustering with PCA visualization
- Geo-experiment design for Test/Control store matching

Case Study: Suburban Discount Retailer (647 stores, 30+ states)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from math import radians, sin, cos, sqrt, atan2

# ML imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="Store Segmentation", page_icon="👥", layout="wide")

# ============================================
# DATA LOADING
# ============================================
@st.cache_data
def load_store_data():
    """Load store segmentation data from CSV"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    data_path = os.path.join(parent_dir, 'data_sources', 'store_segmentation_data.csv')
    
    df = pd.read_csv(data_path)
    return df

# ============================================
# CURATED FEATURES (Business-relevant for discount retailer)
# ============================================
FEATURE_COLS = [
    'TOTAL_POPULATION',
    'population_density',
    'N_ZIPCODES',
    'area_sq_mi',
    'quality_not_price_disagree',
    'price_not_brands_disagree',
    'compare_prices_disagree',
    'buy_american_agree',
    'mail_order_1_49',
    'mail_order_50_99',
    'mail_phone_1_49',
    'mail_phone_500_plus',
    'reads_sunday_newspaper',
    'reads_classifieds',
    'commute_1_2_hrs',
    'work_from_home',
    'public_transit',
]

FEATURE_DESCRIPTIONS = {
    'TOTAL_POPULATION': 'Trade area population',
    'population_density': 'Population per sq mile',
    'N_ZIPCODES': 'Number of zip codes in trade area',
    'area_sq_mi': 'Trade area size (sq miles)',
    'quality_not_price_disagree': 'Disagree: "Buy quality not price" → Price-focused shoppers',
    'price_not_brands_disagree': 'Disagree: "Buy price not brands" → Brand-focused (not target)',
    'compare_prices_disagree': 'Disagree: "Compare prices online"',
    'buy_american_agree': 'Agree: "Buying American is important"',
    'mail_order_1_49': 'Spent $1-49 on mail/phone/internet orders',
    'mail_order_50_99': 'Spent $50-99 on mail/phone/internet orders',
    'mail_phone_1_49': 'Spent $1-49 on mail/phone orders',
    'mail_phone_500_plus': 'Spent $500+ on mail/phone orders',
    'reads_sunday_newspaper': 'Reads 2+ Sunday newspapers',
    'reads_classifieds': 'Reads classified section',
    'commute_1_2_hrs': 'Commutes 1-2 hours/week',
    'work_from_home': 'Works from home',
    'public_transit': 'Uses public transportation',
}

# ============================================
# HELPER FUNCTIONS
# ============================================
def haversine_miles(lat1, lon1, lat2, lon2):
    """Calculate distance in miles between two lat/lon points"""
    R = 3959
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

@st.cache_data
def build_distance_matrix(lats, lons):
    """Build pairwise distance matrix between all stores"""
    n = len(lats)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = haversine_miles(lats[i], lons[i], lats[j], lons[j])
            distances[i, j] = d
            distances[j, i] = d
    return distances

# ============================================
# PAGE CONTENT
# ============================================

st.title("👥 Store Segmentation & Performance Analysis")
st.markdown("*Case Study: Suburban Discount Retailer - Predicting market efficiency using demographic & behavioral features*")

st.markdown("---")

# Load data
try:
    df = load_store_data()
    data_loaded = True
except Exception as e:
    st.error(f"Error loading data: {e}")
    data_loaded = False

if data_loaded:
    
    # Get available features
    available_features = [f for f in FEATURE_COLS if f in df.columns]
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🗺️ Data & Map",
        "🎯 Performance Model",
        "📊 SHAP Analysis",
        "🔬 Store Clustering",
        "🧪 Geo-Experiments"
    ])
    
    # ============================================
    # TAB 1: Data & Map
    # ============================================
    with tab1:
        st.header("Store Network Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Stores", len(df))
        with col2:
            st.metric("States Covered", df['state'].nunique())
        with col3:
            st.metric("Avg Weekly Revenue", f"${df['avg_weekly_revenue'].mean():,.0f}")
        with col4:
            st.metric("Avg Revenue/Capita", f"${df['revenue_per_capita'].mean():.1f}")
        
        st.subheader("Interactive Store Map")
        
        color_by = st.selectbox(
            "Color stores by:",
            ['revenue_per_capita', 'avg_weekly_revenue', 'TOTAL_POPULATION', 'geo_tier'],
            index=0
        )
        
        fig_map = px.scatter_mapbox(
            df,
            lat='lat',
            lon='lon',
            color=color_by,
            size='avg_weekly_revenue',
            hover_name='store_id',
            hover_data=['state', 'geo_tier', 'avg_weekly_revenue', 'revenue_per_capita'],
            color_continuous_scale='RdYlGn' if 'revenue' in color_by else 'Viridis',
            zoom=3,
            height=500,
            title=f"Store Locations (colored by {color_by})"
        )
        fig_map.update_layout(mapbox_style="carto-positron")
        st.plotly_chart(fig_map, use_container_width=True)
        
        # Data summary
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Stores by State (Top 15)")
            state_counts = df['state'].value_counts().head(15)
            fig_states = px.bar(
                x=state_counts.values, 
                y=state_counts.index,
                orientation='h',
                labels={'x': 'Number of Stores', 'y': 'State'}
            )
            fig_states.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_states, use_container_width=True)
        
        with col2:
            st.subheader("Revenue per Capita by Geo Tier")
            geo_revenue = df.groupby('geo_tier')['revenue_per_capita'].mean().sort_values(ascending=True)
            fig_geo = px.bar(
                x=geo_revenue.values,
                y=geo_revenue.index,
                orientation='h',
                labels={'x': 'Avg Revenue per Capita ($)', 'y': 'Geo Tier'},
                color=geo_revenue.values,
                color_continuous_scale='RdYlGn'
            )
            fig_geo.update_layout(height=400)
            st.plotly_chart(fig_geo, use_container_width=True)
        
        # Key insight callout
        st.info("""
        💡 **Key Insight:** Notice how revenue per capita varies by geo tier. 
        Suburban and rural stores often show higher efficiency because they face less competition 
        from major big-box retailers.
        """)
        
        # Feature descriptions
        with st.expander("📋 Feature Descriptions"):
            for feat, desc in FEATURE_DESCRIPTIONS.items():
                if feat in available_features:
                    st.markdown(f"**{feat}**: {desc}")
    
    # ============================================
    # TAB 2: Performance Model (XGBoost)
    # ============================================
    with tab2:
        st.header("Store Performance Prediction")
        
        st.markdown("""
        **Target:** Revenue per Capita ($ per 1,000 trade area residents)
        
        *This measures market EFFICIENCY - how well a store captures its local market, 
        independent of market size.*
        
        **Model:** XGBoost Regressor with 17 curated business-relevant features
        """)
        
        # Import XGBoost
        try:
            import xgboost as xgb
            xgb_available = True
        except ImportError:
            xgb_available = False
            st.warning("XGBoost not installed. Using sklearn GradientBoostingRegressor.")
            from sklearn.ensemble import GradientBoostingRegressor
        
        # Prepare data
        target = 'revenue_per_capita'
        df_model = df.dropna(subset=[target] + available_features)
        
        X = df_model[available_features].fillna(0)
        y = df_model[target]
        
        st.write(f"**Features:** {len(available_features)} | **Samples:** {len(X)}")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if st.button("🚀 Train XGBoost Model", type="primary"):
            with st.spinner("Training model... it will take 3 minutes - due to streamlit CPU limitations"):
                
                if xgb_available:
                    model = xgb.XGBRegressor(
                        n_estimators=100,  # Reduced from 200 for faster training
                        max_depth=4,
                        learning_rate=0.1,  # Increased from 0.05 for faster convergence
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=42,
                        n_jobs=-1  # Use all cores
                    )
                else:
                    model = GradientBoostingRegressor(
                        n_estimators=200,
                        max_depth=4,
                        learning_rate=0.05,
                        random_state=42
                    )
                
                model.fit(X_train, y_train)
                
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                train_r2 = r2_score(y_train, y_pred_train)
                test_r2 = r2_score(y_test, y_pred_test)
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                test_mae = mean_absolute_error(y_test, y_pred_test)
                cv_scores = cross_val_score(model, X, y, cv=3, scoring='r2')  # Reduced from 5 for speed
                
                st.session_state['xgb_model'] = model
                st.session_state['X_train'] = X_train
                st.session_state['X_test'] = X_test
                st.session_state['y_test'] = y_test
                st.session_state['y_pred_test'] = y_pred_test
                st.session_state['feature_cols'] = available_features
            
            st.success("✅ Model trained successfully!")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Train R²", f"{train_r2:.3f}")
            with col2:
                st.metric("Test R²", f"{test_r2:.3f}")
            with col3:
                st.metric("Test MAE", f"${test_mae:.2f}")
            with col4:
                st.metric("CV R²", f"{cv_scores.mean():.3f} ± {cv_scores.std():.2f}")
            
            # Feature importance
            st.subheader("Feature Importance")
            
            importance_df = pd.DataFrame({
                'Feature': available_features,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=True)
            
            fig_imp = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Feature Importance (XGBoost Gain)',
                color='Importance',
                color_continuous_scale='Reds'
            )
            fig_imp.update_layout(height=500)
            st.plotly_chart(fig_imp, use_container_width=True)
            
            # Actual vs Predicted
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Actual vs Predicted")
                fig_scatter = px.scatter(
                    x=y_test,
                    y=y_pred_test,
                    labels={'x': 'Actual Revenue/Capita', 'y': 'Predicted'},
                    title=f'R² = {test_r2:.3f}'
                )
                min_val, max_val = min(y_test.min(), min(y_pred_test)), max(y_test.max(), max(y_pred_test))
                fig_scatter.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                                  mode='lines', name='Perfect', line=dict(dash='dash', color='red')))
                fig_scatter.update_layout(height=400)
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            with col2:
                st.subheader("Residuals")
                residuals = y_test - y_pred_test
                fig_resid = px.histogram(residuals, nbins=30, title='Residual Distribution')
                fig_resid.update_layout(height=400)
                st.plotly_chart(fig_resid, use_container_width=True)
            
            # Business interpretation
            st.subheader("📊 Key Insights")
            
            top_3 = importance_df.tail(3)['Feature'].tolist()[::-1]
            st.markdown(f"""
            **Top 3 Predictors of Market Efficiency:**
            1. **{top_3[0]}**: {FEATURE_DESCRIPTIONS.get(top_3[0], '')}
            2. **{top_3[1]}**: {FEATURE_DESCRIPTIONS.get(top_3[1], '')}
            3. **{top_3[2]}**: {FEATURE_DESCRIPTIONS.get(top_3[2], '')}
            """)
            
            # Model performance note
            if train_r2 > test_r2 + 0.1:
                st.info(f"""
                ℹ️ **Note on Model Performance:**
                - Train R² ({train_r2:.3f}) > Test R² ({test_r2:.3f}) indicates slight overfitting
                - This is expected with {len(available_features)} features and {len(X)} samples
                - **CV R² ({cv_scores.mean():.3f})** is the most conservative/honest estimate of true predictive power
                """)
            
            # Feature interpretation guide
            with st.expander("📖 Feature Interpretation Guide"):
                st.markdown("""
                | Feature | What it measures | Business meaning |
                |---------|------------------|------------------|
                | **TOTAL_POPULATION** | Trade area population | Smaller markets = less competition = higher efficiency |
                | **N_ZIPCODES** | Zip codes in trade area | More zips = larger geographic spread (suburban/rural) |
                | **commute_1_2_hrs** | 1-2 hour weekly commuters | Suburban workers who drive = core demographic |
                | **population_density** | People per sq mile | Lower density = suburban/rural = retailer's sweet spot |
                | **mail_order_1_49** | $1-49 mail/catalog spenders | Mail-responsive = responds to print flyers! |
                | **mail_order_50_99** | $50-99 mail spenders | Higher mail engagement |
                | **mail_phone_500_plus** | $500+ mail/phone spenders | Premium mail-order customers |
                | **reads_sunday_newspaper** | Sunday newspaper readers | Traditional media consumers |
                | **reads_classifieds** | Classified section readers | Deal-seekers, traditional shoppers |
                | **quality_not_price_disagree** | Disagrees "buy quality not price" | Price-focused shoppers = core customer! |
                | **price_not_brands_disagree** | Disagrees "buy price not brands" | Brand-focused (NOT target customer) |
                | **work_from_home** | Remote workers | Lifestyle indicator |
                | **public_transit** | Public transit users | Urban indicator (not primary target) |
                """)
        else:
            st.info("👆 Click the button above to train the model")
    
    # ============================================
    # TAB 3: SHAP Analysis
    # ============================================
    with tab3:
        st.header("SHAP Explainability")
        
        st.markdown("""
        **SHAP** (SHapley Additive exPlanations) shows:
        - **Global:** Which features matter most across all stores
        - **Local:** Why a specific store has high/low predicted performance
        """)
        
        if 'xgb_model' not in st.session_state:
            st.warning("⚠️ Please train the model in the 'Performance Model' tab first.")
        else:
            try:
                import shap
                shap_available = True
            except ImportError:
                shap_available = False
                st.error("SHAP library not installed. Run: `pip install shap`")
            
            if shap_available:
                model = st.session_state['xgb_model']
                X_train = st.session_state['X_train']
                X_test = st.session_state['X_test']
                feature_cols = st.session_state['feature_cols']
                
                if st.button("📊 Generate SHAP Analysis", type="primary"):
                    with st.spinner("Computing SHAP values...it takes 2 min"):
                        
                        X_sample = X_test.sample(min(50, len(X_test)), random_state=42)  # Reduced from 100
                        X_background = X_train.sample(min(30, len(X_train)), random_state=42)  # Reduced from 50
                        
                        try:
                            explainer = shap.Explainer(model.predict, X_background)
                            shap_values_obj = explainer(X_sample)
                            shap_values = shap_values_obj.values
                            base_value = float(shap_values_obj.base_values.mean())
                            
                            st.session_state['shap_values'] = shap_values
                            st.session_state['X_sample'] = X_sample
                            st.session_state['base_value'] = base_value
                            
                            st.success("✅ SHAP values computed!")
                        except Exception as e:
                            st.error(f"SHAP computation failed: {str(e)[:200]}")
                
                if 'shap_values' in st.session_state and st.session_state['shap_values'] is not None:
                    shap_values = st.session_state['shap_values']
                    X_sample = st.session_state['X_sample']
                    
                    st.subheader("Global Feature Importance (SHAP)")
                    
                    mean_abs_shap = np.abs(shap_values).mean(axis=0)
                    shap_importance = pd.DataFrame({
                        'Feature': feature_cols,
                        'Mean |SHAP|': mean_abs_shap
                    }).sort_values('Mean |SHAP|', ascending=True)
                    
                    fig_shap = px.bar(
                        shap_importance,
                        x='Mean |SHAP|',
                        y='Feature',
                        orientation='h',
                        title='Mean |SHAP| Value (Impact on Revenue per Capita)',
                        color='Mean |SHAP|',
                        color_continuous_scale='Reds'
                    )
                    fig_shap.update_layout(height=500)
                    st.plotly_chart(fig_shap, use_container_width=True)
                    
                    # Beeswarm
                    st.subheader("SHAP Summary Plot")
                    st.markdown("*Red = high feature value, Blue = low feature value*")
                    
                    with st.expander("📖 How to Read This Plot"):
                        st.markdown("""
                        **Axes:**
                        - **X-axis:** SHAP value (RIGHT = increases prediction, LEFT = decreases)
                        - **Y-axis:** Features ranked by importance
                        
                        **Colors:**
                        - 🔴 **Red dots** = HIGH feature value
                        - 🔵 **Blue dots** = LOW feature value
                        
                        **Example Patterns:**
                        - `TOTAL_POPULATION`: Blue dots → RIGHT means LOW population → HIGHER revenue/capita
                        - `reads_classifieds`: Red dots → RIGHT means HIGH readership → HIGHER revenue/capita
                        """)
                    
                    top_features = shap_importance.tail(10)['Feature'].tolist()
                    
                    fig_beeswarm = go.Figure()
                    
                    for i, feat in enumerate(reversed(top_features)):
                        feat_idx = feature_cols.index(feat)
                        feat_shap = shap_values[:, feat_idx]
                        feat_values = X_sample[feat].values
                        feat_norm = (feat_values - feat_values.min()) / (feat_values.max() - feat_values.min() + 1e-10)
                        
                        fig_beeswarm.add_trace(go.Scatter(
                            x=feat_shap,
                            y=[i + np.random.uniform(-0.2, 0.2) for _ in range(len(feat_shap))],
                            mode='markers',
                            marker=dict(size=6, color=feat_norm, colorscale='RdBu_r', showscale=(i == 0)),
                            name=feat,
                            showlegend=False,
                            hovertemplate=f'{feat}<br>SHAP: %{{x:.1f}}<br>Value: %{{marker.color:.0f}}<extra></extra>'
                        ))
                    
                    fig_beeswarm.update_layout(
                        height=450,
                        title='SHAP Values by Feature',
                        xaxis_title='SHAP Value (impact on prediction)',
                        yaxis=dict(tickmode='array', tickvals=list(range(len(top_features))),
                                   ticktext=list(reversed(top_features)))
                    )
                    fig_beeswarm.add_vline(x=0, line_dash='dash', line_color='gray')
                    st.plotly_chart(fig_beeswarm, use_container_width=True)
                    
                    # Waterfall
                    st.subheader("Individual Store Explanation")
                    
                    store_idx = st.selectbox(
                        "Select store to explain:",
                        range(len(X_sample)),
                        format_func=lambda x: f"Store {X_sample.index[x]} - Predicted: ${model.predict(X_sample.iloc[[x]])[0]:.1f}/capita"
                    )
                    
                    store_shap = shap_values[store_idx]
                    base_value = st.session_state['base_value']
                    
                    sorted_idx = np.argsort(np.abs(store_shap))[::-1][:10]
                    
                    waterfall_df = pd.DataFrame({
                        'Feature': [feature_cols[i] for i in sorted_idx],
                        'SHAP Value': [store_shap[i] for i in sorted_idx],
                        'Feature Value': [X_sample.iloc[store_idx, i] for i in sorted_idx]
                    })
                    
                    fig_waterfall = go.Figure(go.Waterfall(
                        name="SHAP",
                        orientation="h",
                        y=waterfall_df['Feature'].tolist()[::-1],
                        x=waterfall_df['SHAP Value'].tolist()[::-1],
                        connector={"line": {"color": "rgb(63, 63, 63)"}},
                        decreasing={"marker": {"color": "#EF553B"}},
                        increasing={"marker": {"color": "#00CC96"}},
                        base=base_value
                    ))
                    
                    fig_waterfall.update_layout(
                        title=f'SHAP Waterfall - Store {X_sample.index[store_idx]}',
                        height=400,
                        xaxis_title='Revenue per Capita Impact ($)'
                    )
                    st.plotly_chart(fig_waterfall, use_container_width=True)
                    
                    st.dataframe(waterfall_df.round(2), use_container_width=True, hide_index=True)
                    
                    # Business interpretation
                    st.subheader("🎯 Business Interpretation")
                    
                    st.markdown("""
                    ### How to Read SHAP Values
                    
                    | SHAP Direction | Meaning |
                    |----------------|---------|
                    | **Positive (+)** | Feature INCREASES predicted revenue per capita |
                    | **Negative (-)** | Feature DECREASES predicted revenue per capita |
                    """)
                    
                    st.markdown("---")
                    
                    st.markdown("""
                    ### Key Insights for Suburban Discount Retailer
                    
                    #### 1. 🏘️ **TOTAL_POPULATION (Negative Relationship)**
                    - **Pattern:** Smaller markets → Higher revenue per capita
                    - **Why:** Less competition from major big-box retailers
                    - **Strategy:** The retailer wins in underserved, smaller markets
                    
                    #### 2. 🌆 **Population Density (Negative Relationship)**
                    - **Pattern:** Lower density → Higher efficiency
                    - **Why:** Suburban/rural areas have fewer retail options
                    - **Strategy:** Confirms the suburban/rural site selection strategy
                    
                    #### 3. 📬 **Mail Order Spending (Positive Relationship)**
                    - **Pattern:** Mail-responsive consumers → Higher revenue
                    - **Features:** `mail_order_1_49`, `mail_order_50_99`, `mail_phone_500_plus`
                    - **Why:** People who respond to mail/catalogs also respond to **print flyers**
                    - **Strategy:** Validates the **30% print flyer advertising investment!**
                    
                    #### 4. 📰 **Newspaper Readers (Positive Relationship)**
                    - **Pattern:** Traditional media consumers → Higher revenue
                    - **Features:** `reads_sunday_newspaper`, `reads_classifieds`
                    - **Why:** Deal-seekers who read classifieds are bargain hunters
                    - **Strategy:** Target customer is traditional, not digital-first
                    
                    #### 5. 🚗 **Commuters (Positive Relationship)**
                    - **Pattern:** 1-2 hour weekly commuters → Higher revenue
                    - **Why:** Suburban workers who drive = core demographic
                    - **Strategy:** Locate stores along commuter routes
                    """)
                    
                    st.markdown("---")
                    
                    st.success("""
                    **📊 Executive Summary:**
                    
                    This suburban discount retailer performs best in **smaller, lower-density suburban markets** with 
                    **traditional, mail-responsive consumers** who read newspapers and commute by car.
                    
                    This validates:
                    - ✅ Suburban/rural site selection strategy
                    - ✅ 30% print flyer advertising spend
                    - ✅ Focus on underserved markets
                    """)
    
    # ============================================
    # TAB 4: Store Clustering
    # ============================================
    with tab4:
        st.header("Store Clustering")
        
        st.markdown("""
        **Objective:** Identify natural groupings of stores based on market characteristics.
        
        **Method:** K-Means clustering with PCA visualization.
        """)
        
        clustering_features = [f for f in available_features if f in df.columns]
        
        df_cluster = df.dropna(subset=clustering_features + ['revenue_per_capita']).copy()
        X_cluster = df_cluster[clustering_features].fillna(0)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_cluster)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            n_clusters = st.slider("Number of Clusters (K)", 2, 8, 4)
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)  # Reduced for speed
            clusters = kmeans.fit_predict(X_scaled)
            df_cluster['Cluster'] = clusters
            df_cluster['Cluster_str'] = df_cluster['Cluster'].astype(str)
            
            sil_score = silhouette_score(X_scaled, clusters)
            st.metric("Silhouette Score", f"{sil_score:.3f}")
        
        with col2:
            # Elbow plot with caching
            inertias = []
            for k in range(2, 8):  # Reduced from 10 for speed
                km = KMeans(n_clusters=k, random_state=42, n_init=5)  # Reduced n_init from 10
                km.fit(X_scaled)
                inertias.append(km.inertia_)
            
            fig_elbow = px.line(x=list(range(2, 8)), y=inertias, markers=True,
                               labels={'x': 'K', 'y': 'Inertia'}, title='Elbow Method')
            fig_elbow.add_vline(x=n_clusters, line_dash='dash', line_color='red')
            fig_elbow.update_layout(height=250)
            st.plotly_chart(fig_elbow, use_container_width=True)
        
        # PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        df_cluster['PC1'] = X_pca[:, 0]
        df_cluster['PC2'] = X_pca[:, 1]
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_pca = px.scatter(
                df_cluster, x='PC1', y='PC2', color='Cluster_str',
                hover_data=['store_id', 'state', 'revenue_per_capita'],
                title=f'Store Clusters (K={n_clusters})',
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            fig_pca.update_traces(marker=dict(size=8))
            fig_pca.update_layout(height=400)
            st.plotly_chart(fig_pca, use_container_width=True)
        
        with col2:
            fig_pca_perf = px.scatter(
                df_cluster, x='PC1', y='PC2', color='revenue_per_capita',
                hover_data=['store_id', 'Cluster'],
                title='Colored by Revenue/Capita',
                color_continuous_scale='RdYlGn'
            )
            fig_pca_perf.update_traces(marker=dict(size=8))
            fig_pca_perf.update_layout(height=400)
            st.plotly_chart(fig_pca_perf, use_container_width=True)
        
        # Cluster profiles
        st.subheader("Cluster Profiles")
        
        profile_cols = ['revenue_per_capita', 'avg_weekly_revenue', 'TOTAL_POPULATION', 
                        'population_density', 'N_ZIPCODES']
        profile_cols = [c for c in profile_cols if c in df_cluster.columns]
        
        cluster_profiles = df_cluster.groupby('Cluster')[profile_cols].mean().round(1)
        cluster_profiles['Store Count'] = df_cluster.groupby('Cluster').size()
        
        st.dataframe(cluster_profiles, use_container_width=True)
        
        # Cluster interpretation
        st.markdown("---")
        st.subheader("🔍 Cluster Interpretation Guide")
        st.markdown("""
        **How to use clusters for business decisions:**
        
        | Cluster Profile | Typical Characteristics | Marketing Strategy |
        |-----------------|------------------------|-------------------|
        | High revenue/capita + Low population | Small, efficient markets | Protect & maintain |
        | High revenue/capita + High population | Large, successful markets | Maximize investment |
        | Low revenue/capita + Low population | Struggling small markets | Evaluate for closure |
        | Low revenue/capita + High population | Underperforming large markets | Investigate competition |
        
        **Use Cases:**
        - 🎯 **Targeted Marketing:** Different messaging per cluster
        - 📍 **Site Selection:** Find markets similar to top-performing clusters
        - 💰 **Resource Allocation:** Invest more in high-potential clusters
        """)
        
        # Map
        st.subheader("Geographic Distribution")
        
        fig_cluster_map = px.scatter_mapbox(
            df_cluster, lat='lat', lon='lon', color='Cluster_str',
            size='avg_weekly_revenue', hover_name='store_id',
            hover_data=['state', 'revenue_per_capita'],
            zoom=3, height=450, title='Store Clusters on Map',
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        fig_cluster_map.update_layout(mapbox_style="carto-positron")
        st.plotly_chart(fig_cluster_map, use_container_width=True)
    
    # ============================================
    # TAB 5: Geo-Experiments
    # ============================================
    with tab5:
        st.header("Geo-Experiment Design: Test/Control Matching")
        
        st.markdown("""
        **Business Problem:** Design a geo-experiment where:
        1. **Test stores** receive treatment (e.g., new ad campaign)
        2. **Control stores** are demographically similar but geographically separated
        
        **Method:** Propensity matching + minimum distance constraint to avoid spillover
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            n_pairs = st.slider("Number of Test/Control Pairs", 10, 100, 50)
        with col2:
            min_distance = st.slider("Minimum Distance (miles)", 10, 100, 25)
        
        match_features = ['TOTAL_POPULATION', 'population_density', 'revenue_per_capita', 
                         'quality_not_price_disagree', 'reads_sunday_newspaper']
        match_features = [f for f in match_features if f in df.columns]
        
        if st.button("🔬 Generate Test/Control Pairs", type="primary"):
            with st.spinner("Finding optimal store pairs..."):
                
                df_geo = df.dropna(subset=['lat', 'lon'] + match_features).copy().reset_index(drop=True)
                
                scaler = StandardScaler()
                X_match = scaler.fit_transform(df_geo[match_features].fillna(0))
                
                geo_distances = build_distance_matrix(df_geo['lat'].values, df_geo['lon'].values)
                
                nn = NearestNeighbors(n_neighbors=min(50, len(df_geo))).fit(X_match)  # Reduced from 100
                feature_distances, indices = nn.kneighbors(X_match)
                
                pairs = []
                used = set()
                
                for i in range(len(df_geo)):
                    if i in used:
                        continue
                    for j_idx in range(1, len(indices[i])):
                        j = indices[i][j_idx]
                        if j in used:
                            continue
                        if geo_distances[i, j] >= min_distance:
                            pairs.append({
                                'test_store': int(df_geo.iloc[i]['store_id']),
                                'test_state': df_geo.iloc[i]['state'],
                                'control_store': int(df_geo.iloc[j]['store_id']),
                                'control_state': df_geo.iloc[j]['state'],
                                'similarity': 1 / (1 + feature_distances[i][j_idx]),
                                'distance_miles': round(geo_distances[i, j], 1),
                                'test_lat': df_geo.iloc[i]['lat'],
                                'test_lon': df_geo.iloc[i]['lon'],
                                'control_lat': df_geo.iloc[j]['lat'],
                                'control_lon': df_geo.iloc[j]['lon'],
                            })
                            used.add(i)
                            used.add(j)
                            break
                    if len(pairs) >= n_pairs:
                        break
                
                pairs_df = pd.DataFrame(pairs)
            
            st.success(f"✅ Found {len(pairs_df)} matched pairs!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Pairs Found", len(pairs_df))
            with col2:
                st.metric("Avg Distance", f"{pairs_df['distance_miles'].mean():.1f} mi")
            with col3:
                st.metric("Avg Similarity", f"{pairs_df['similarity'].mean():.3f}")
            
            st.dataframe(pairs_df[['test_store', 'test_state', 'control_store', 'control_state',
                                   'distance_miles', 'similarity']].round(3), 
                        use_container_width=True, hide_index=True)
            
            # Map
            fig_pairs = go.Figure()
            
            for _, row in pairs_df.iterrows():
                fig_pairs.add_trace(go.Scattermapbox(
                    lat=[row['test_lat'], row['control_lat']],
                    lon=[row['test_lon'], row['control_lon']],
                    mode='lines', line=dict(width=1, color='gray'),
                    showlegend=False, hoverinfo='skip'
                ))
            
            fig_pairs.add_trace(go.Scattermapbox(
                lat=pairs_df['test_lat'], lon=pairs_df['test_lon'],
                mode='markers', marker=dict(size=10, color='red'),
                name='Test Stores', text=pairs_df['test_store']
            ))
            fig_pairs.add_trace(go.Scattermapbox(
                lat=pairs_df['control_lat'], lon=pairs_df['control_lon'],
                mode='markers', marker=dict(size=10, color='blue'),
                name='Control Stores', text=pairs_df['control_store']
            ))
            
            fig_pairs.update_layout(
                mapbox_style="carto-positron",
                mapbox=dict(center=dict(lat=39, lon=-95), zoom=3),
                height=500, title=f'Test/Control Pairs (Min: {min_distance} miles)'
            )
            st.plotly_chart(fig_pairs, use_container_width=True)
            
            st.download_button(
                "📥 Download Pairs CSV",
                pairs_df.to_csv(index=False),
                "test_control_pairs.csv",
                "text/csv"
            )
            
            # Interpretation
            st.markdown("---")
            st.subheader("🔬 How to Use These Pairs")
            st.markdown(f"""
            **Experiment Design:**
            1. **Test Stores** (red): Receive the treatment (e.g., new TV campaign, increased flyer frequency)
            2. **Control Stores** (blue): No treatment change (baseline comparison)
            3. **Minimum Distance:** {min_distance} miles ensures no ad spillover between pairs
            
            **Measuring Incrementality:**
            ```
            Incremental Lift = (Test Store Sales - Control Store Sales) / Control Store Sales
            ```
            
            **Why This Matters:**
            - Avoids contamination from overlapping media markets
            - Demographically similar stores isolate the treatment effect
            - Enables causal inference (not just correlation)
            
            **Best Practices:**
            - Run experiment for 4-8 weeks minimum
            - Monitor both stores for external factors (weather, competition)
            - Use difference-in-differences analysis for statistical rigor
            """)
    
    # ============================================
    # METHODOLOGY
    # ============================================
    with st.expander("📋 Methodology & Technical Details"):
        st.markdown("""
        ## Data Sources
        - **647 retail stores** across 30+ US states
        - **ESRI demographic data** (315 features) matched by zip code
        - **Sales data** aggregated to weekly averages
        
        ## Target Variable
        **Revenue per Capita** ($ per 1,000 trade area residents)
        
        *Why this target?*
        - Measures market EFFICIENCY, not just store size
        - Normalizes for population differences
        - More predictable than absolute revenue (R² 0.82 vs 0.03)
        
        ## Feature Selection Process
        1. Started with **315 ESRI features**
        2. Removed zero-variance features
        3. Removed highly correlated features (>0.98)
        4. Used XGBoost feature importance to identify top predictors
        5. Curated to **17 business-relevant features**
        
        ## Model Architecture
        ```
        XGBoost Regressor
        - n_estimators: 100
        - max_depth: 4
        - learning_rate: 0.1
        - subsample: 0.8
        - colsample_bytree: 0.8
        ```
        
        ## Performance Metrics
        | Metric | Value | Interpretation |
        |--------|-------|----------------|
        | Train R² | 0.987 | Model fits training data well |
        | Test R² | 0.821 | Strong generalization |
        | CV R² | 0.639 ± 0.20 | Conservative estimate (3-fold) |
        | MAE | $9.24 | Average prediction error |
        
        ## Key Findings
        
        ### Markets Where the Retailer Performs Best:
        1. **Smaller population** - Less competition
        2. **Lower density** - Suburban/rural preference
        3. **Mail-responsive** - Validates print flyer strategy
        4. **Traditional media consumers** - Newspaper readers
        5. **Car commuters** - Suburban lifestyle
        
        ### Business Implications:
        - ✅ **Site Selection:** Prioritize underserved suburban markets
        - ✅ **Advertising:** Continue 30% print flyer investment
        - ✅ **Target Customer:** Traditional, deal-seeking, suburban shoppers
        
        ## SHAP Explainability
        - Uses `shap.Explainer` with model predict function
        - Background sample: 30 stores
        - Explanation sample: 50 stores
        - Provides both global (feature importance) and local (individual store) explanations
        """)
