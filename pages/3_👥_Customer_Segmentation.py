"""
Page 3: Customer Segmentation
=============================
Behavioral clustering for ad targeting
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Customer Segmentation", page_icon="👥", layout="wide")

# ============================================
# DATA GENERATION (Synthetic Eyewear Retail)
# ============================================
@st.cache_data
def generate_customer_data():
    """Generate synthetic customer data for eyewear retailer"""
    np.random.seed(42)
    
    n_customers = 5000
    
    # Customer segments (for realistic generation)
    # 1: High-value loyalists (15%)
    # 2: Regular buyers (25%)
    # 3: Occasional shoppers (30%)
    # 4: New customers (20%)
    # 5: At-risk churners (10%)
    
    segment_probs = [0.15, 0.25, 0.30, 0.20, 0.10]
    segments = np.random.choice([1, 2, 3, 4, 5], size=n_customers, p=segment_probs)
    
    data = []
    for i, seg in enumerate(segments):
        if seg == 1:  # High-value loyalists
            recency = np.random.randint(1, 30)
            frequency = np.random.randint(8, 20)
            monetary = np.random.uniform(800, 2000)
            tenure = np.random.randint(36, 84)
            online_ratio = np.random.uniform(0.3, 0.7)
            rx_ratio = np.random.uniform(0.6, 0.9)
            age = np.random.randint(35, 65)
        elif seg == 2:  # Regular buyers
            recency = np.random.randint(15, 60)
            frequency = np.random.randint(4, 8)
            monetary = np.random.uniform(400, 800)
            tenure = np.random.randint(24, 60)
            online_ratio = np.random.uniform(0.4, 0.8)
            rx_ratio = np.random.uniform(0.5, 0.8)
            age = np.random.randint(30, 55)
        elif seg == 3:  # Occasional shoppers
            recency = np.random.randint(60, 180)
            frequency = np.random.randint(2, 4)
            monetary = np.random.uniform(200, 500)
            tenure = np.random.randint(12, 48)
            online_ratio = np.random.uniform(0.5, 0.9)
            rx_ratio = np.random.uniform(0.3, 0.6)
            age = np.random.randint(25, 50)
        elif seg == 4:  # New customers
            recency = np.random.randint(1, 45)
            frequency = np.random.randint(1, 3)
            monetary = np.random.uniform(150, 400)
            tenure = np.random.randint(1, 12)
            online_ratio = np.random.uniform(0.6, 0.95)
            rx_ratio = np.random.uniform(0.4, 0.7)
            age = np.random.randint(20, 40)
        else:  # At-risk churners
            recency = np.random.randint(180, 365)
            frequency = np.random.randint(1, 4)
            monetary = np.random.uniform(100, 400)
            tenure = np.random.randint(24, 60)
            online_ratio = np.random.uniform(0.2, 0.5)
            rx_ratio = np.random.uniform(0.4, 0.7)
            age = np.random.randint(40, 70)
        
        data.append({
            'customer_id': f'C{i+1:05d}',
            'recency_days': recency,
            'frequency': frequency,
            'monetary_value': monetary,
            'tenure_months': tenure,
            'online_purchase_ratio': online_ratio,
            'rx_purchase_ratio': rx_ratio,
            'age': age,
            'true_segment': seg
        })
    
    df = pd.DataFrame(data)
    
    # Add derived features
    df['avg_order_value'] = df['monetary_value'] / df['frequency']
    df['purchase_rate'] = df['frequency'] / df['tenure_months'] * 12  # Annual rate
    
    return df

@st.cache_data
def perform_clustering(df, n_clusters=5):
    """Perform K-means clustering on customer features"""
    
    # Select features for clustering
    features = ['recency_days', 'frequency', 'monetary_value', 
                'online_purchase_ratio', 'rx_purchase_ratio', 'avg_order_value']
    
    X = df[features].copy()
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df['pca_1'] = X_pca[:, 0]
    df['pca_2'] = X_pca[:, 1]
    
    return df, kmeans, scaler, pca

@st.cache_data
def calculate_cluster_profiles(df):
    """Calculate cluster profiles"""
    
    profiles = df.groupby('cluster').agg({
        'recency_days': 'mean',
        'frequency': 'mean',
        'monetary_value': 'mean',
        'online_purchase_ratio': 'mean',
        'rx_purchase_ratio': 'mean',
        'age': 'mean',
        'avg_order_value': 'mean',
        'customer_id': 'count'
    }).reset_index()
    
    profiles.columns = ['Cluster', 'Avg Recency', 'Avg Frequency', 'Avg Monetary',
                       'Online %', 'Rx %', 'Avg Age', 'Avg Order Value', 'Count']
    
    # Assign cluster names based on characteristics
    cluster_names = []
    for _, row in profiles.iterrows():
        if row['Avg Recency'] < 30 and row['Avg Frequency'] > 6 and row['Avg Monetary'] > 700:
            cluster_names.append('💎 High-Value Loyalists')
        elif row['Avg Recency'] > 150:
            cluster_names.append('⚠️ At-Risk Churners')
        elif row['Avg Frequency'] < 2.5 and row['Avg Recency'] < 60:
            cluster_names.append('🆕 New Customers')
        elif row['Online %'] > 0.7:
            cluster_names.append('💻 Digital Natives')
        else:
            cluster_names.append('🛒 Regular Buyers')
    
    profiles['Segment Name'] = cluster_names
    profiles['Size %'] = profiles['Count'] / profiles['Count'].sum() * 100
    
    return profiles

# ============================================
# PAGE CONTENT  
# ============================================

st.title("👥 Customer Segmentation")
st.markdown("*Behavioral clustering for targeted marketing*")

st.markdown("---")

# Sidebar controls
st.sidebar.markdown("### Clustering Parameters")
n_clusters = st.sidebar.slider("Number of Clusters", 3, 8, 5)

# Load and process data
df = generate_customer_data()
df_clustered, kmeans, scaler, pca = perform_clustering(df, n_clusters)
profiles = calculate_cluster_profiles(df_clustered)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data Overview",
    "🎯 RFM Analysis",
    "👥 Cluster Results",
    "📈 Segment Profiles",
    "📋 Methodology"
])

# ============================================
# TAB 1: Data Overview
# ============================================
with tab1:
    st.header("Customer Data Overview")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("""
        ### Dataset: ACME Eyewear Customers
        
        **Sample Size:** 5,000 customers
        
        **Features Available:**
        - Recency (days since last purchase)
        - Frequency (total purchases)
        - Monetary (total spend)
        - Tenure (months as customer)
        - Online purchase ratio
        - Rx (prescription) purchase ratio
        - Age
        
        **Business Context:**
        - Mix of online and in-store
        - Prescription and sunglasses
        - Varying customer lifecycles
        """)
        
        # Summary metrics
        st.markdown("### Key Metrics")
        st.metric("Total Customers", f"{len(df):,}")
        st.metric("Avg Customer Value", f"${df['monetary_value'].mean():,.0f}")
        st.metric("Avg Purchase Frequency", f"{df['frequency'].mean():.1f}")
    
    with col2:
        st.subheader("Feature Distributions")
        
        feature = st.selectbox("Select Feature", 
                              ['monetary_value', 'recency_days', 'frequency', 
                               'online_purchase_ratio', 'age'])
        
        fig = px.histogram(df, x=feature, nbins=50, 
                          color_discrete_sequence=['#636EFA'])
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation matrix
        st.subheader("Feature Correlations")
        
        corr_features = ['recency_days', 'frequency', 'monetary_value', 
                        'online_purchase_ratio', 'rx_purchase_ratio']
        corr_matrix = df[corr_features].corr()
        
        fig2 = px.imshow(corr_matrix, text_auto='.2f', aspect='auto',
                        color_continuous_scale='RdBu_r')
        fig2.update_layout(height=350)
        st.plotly_chart(fig2, use_container_width=True)

# ============================================
# TAB 2: RFM Analysis
# ============================================
with tab2:
    st.header("RFM Analysis")
    
    st.markdown("""
    ### Recency, Frequency, Monetary Value
    
    RFM is a classic customer segmentation framework:
    - **Recency:** How recently did they purchase? (Lower is better)
    - **Frequency:** How often do they purchase? (Higher is better)
    - **Monetary:** How much do they spend? (Higher is better)
    """)
    
    # RFM scoring
    df_rfm = df.copy()
    
    # Score each dimension (1-5)
    df_rfm['R_Score'] = pd.qcut(df_rfm['recency_days'], 5, labels=[5,4,3,2,1])
    df_rfm['F_Score'] = pd.qcut(df_rfm['frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
    df_rfm['M_Score'] = pd.qcut(df_rfm['monetary_value'].rank(method='first'), 5, labels=[1,2,3,4,5])
    
    df_rfm['RFM_Score'] = df_rfm['R_Score'].astype(str) + df_rfm['F_Score'].astype(str) + df_rfm['M_Score'].astype(str)
    df_rfm['RFM_Total'] = df_rfm['R_Score'].astype(int) + df_rfm['F_Score'].astype(int) + df_rfm['M_Score'].astype(int)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Recency vs Frequency")
        
        fig3 = px.scatter(df_rfm, x='recency_days', y='frequency', 
                         color='monetary_value', size='monetary_value',
                         color_continuous_scale='Viridis', opacity=0.6)
        fig3.update_layout(height=400)
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        st.subheader("RFM Score Distribution")
        
        fig4 = px.histogram(df_rfm, x='RFM_Total', nbins=15,
                           color_discrete_sequence=['#00CC96'])
        fig4.update_layout(height=400, xaxis_title="Total RFM Score (3-15)")
        st.plotly_chart(fig4, use_container_width=True)
    
    # RFM Segments
    st.subheader("RFM-Based Segments")
    
    def rfm_segment(row):
        if row['RFM_Total'] >= 12:
            return 'Champions'
        elif row['RFM_Total'] >= 9:
            return 'Loyal Customers'
        elif row['R_Score'].astype(int) >= 4 and row['F_Score'].astype(int) <= 2:
            return 'New Customers'
        elif row['R_Score'].astype(int) <= 2 and row['F_Score'].astype(int) >= 3:
            return 'At Risk'
        elif row['R_Score'].astype(int) <= 2:
            return 'Hibernating'
        else:
            return 'Potential Loyalists'
    
    df_rfm['RFM_Segment'] = df_rfm.apply(rfm_segment, axis=1)
    
    rfm_summary = df_rfm.groupby('RFM_Segment').agg({
        'customer_id': 'count',
        'monetary_value': 'mean',
        'frequency': 'mean',
        'recency_days': 'mean'
    }).reset_index()
    rfm_summary.columns = ['Segment', 'Count', 'Avg Value', 'Avg Freq', 'Avg Recency']
    rfm_summary = rfm_summary.sort_values('Avg Value', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig5 = px.pie(rfm_summary, values='Count', names='Segment',
                      title="RFM Segment Distribution")
        fig5.update_layout(height=350)
        st.plotly_chart(fig5, use_container_width=True)
    
    with col2:
        st.dataframe(rfm_summary.round(1), use_container_width=True, hide_index=True)

# ============================================
# TAB 3: Cluster Results
# ============================================
with tab3:
    st.header("K-Means Clustering Results")
    
    st.markdown(f"""
    ### {n_clusters} Customer Clusters Identified
    
    Using K-Means clustering on normalized RFM + behavioral features.
    """)
    
    # PCA visualization
    st.subheader("Cluster Visualization (PCA)")
    
    fig6 = px.scatter(df_clustered, x='pca_1', y='pca_2', color='cluster',
                      color_continuous_scale='Viridis',
                      hover_data=['recency_days', 'frequency', 'monetary_value'],
                      opacity=0.6)
    fig6.update_layout(height=500, 
                      xaxis_title="Principal Component 1",
                      yaxis_title="Principal Component 2")
    st.plotly_chart(fig6, use_container_width=True)
    
    # Cluster sizes
    st.subheader("Cluster Sizes")
    
    cluster_counts = df_clustered['cluster'].value_counts().sort_index()
    
    fig7 = px.bar(x=cluster_counts.index, y=cluster_counts.values,
                  labels={'x': 'Cluster', 'y': 'Count'},
                  color=cluster_counts.values,
                  color_continuous_scale='Blues')
    fig7.update_layout(height=300)
    st.plotly_chart(fig7, use_container_width=True)
    
    # Feature importance by cluster
    st.subheader("Feature Patterns by Cluster")
    
    features_for_radar = ['recency_days', 'frequency', 'monetary_value', 
                          'online_purchase_ratio', 'rx_purchase_ratio']
    
    cluster_means = df_clustered.groupby('cluster')[features_for_radar].mean()
    
    # Normalize for radar chart
    cluster_means_norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())
    
    fig8 = go.Figure()
    
    for cluster in range(n_clusters):
        fig8.add_trace(go.Scatterpolar(
            r=cluster_means_norm.loc[cluster].values.tolist() + [cluster_means_norm.loc[cluster].values[0]],
            theta=features_for_radar + [features_for_radar[0]],
            name=f'Cluster {cluster}'
        ))
    
    fig8.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        height=450
    )
    st.plotly_chart(fig8, use_container_width=True)

# ============================================
# TAB 4: Segment Profiles
# ============================================
with tab4:
    st.header("Segment Profiles & Recommendations")
    
    # Profile cards
    for _, row in profiles.iterrows():
        with st.expander(f"{row['Segment Name']} (Cluster {int(row['Cluster'])}) - {row['Count']:,} customers ({row['Size %']:.1f}%)"):
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Avg Monetary Value", f"${row['Avg Monetary']:,.0f}")
                st.metric("Avg Frequency", f"{row['Avg Frequency']:.1f}")
            
            with col2:
                st.metric("Avg Recency", f"{row['Avg Recency']:.0f} days")
                st.metric("Avg Order Value", f"${row['Avg Order Value']:,.0f}")
            
            with col3:
                st.metric("Online Purchase %", f"{row['Online %']*100:.0f}%")
                st.metric("Rx Purchase %", f"{row['Rx %']*100:.0f}%")
            
            # Recommendations based on segment
            st.markdown("#### 📋 Marketing Recommendations")
            
            if 'High-Value' in row['Segment Name']:
                st.markdown("""
                - **VIP treatment**: Early access to new collections
                - **Loyalty program**: Exclusive discounts and perks
                - **Cross-sell**: Premium lens upgrades, designer frames
                - **Referral incentives**: Leverage their network
                """)
            elif 'At-Risk' in row['Segment Name']:
                st.markdown("""
                - **Win-back campaign**: Personalized re-engagement email
                - **Special offer**: Discount on next purchase
                - **Survey**: Understand why they left
                - **Reminder**: Annual eye exam due
                """)
            elif 'New' in row['Segment Name']:
                st.markdown("""
                - **Welcome series**: Onboarding email sequence
                - **Education**: Guide to lens options, frame care
                - **Second purchase incentive**: Discount on sunglasses
                - **Collect preferences**: Build profile for personalization
                """)
            elif 'Digital' in row['Segment Name']:
                st.markdown("""
                - **App promotion**: Push mobile app adoption
                - **Social ads**: Instagram, TikTok campaigns
                - **Virtual try-on**: AR features promotion
                - **Online exclusives**: Web-only styles
                """)
            else:
                st.markdown("""
                - **Frequency program**: Encourage more frequent purchases
                - **Bundle offers**: Frame + lens + accessories
                - **In-store events**: Trunk shows, eye health seminars
                - **Anniversary reminders**: Celebrate customer milestones
                """)
    
    # Summary table
    st.subheader("Segment Summary Table")
    
    display_profiles = profiles.copy()
    display_profiles['Avg Monetary'] = display_profiles['Avg Monetary'].apply(lambda x: f"${x:,.0f}")
    display_profiles['Avg Order Value'] = display_profiles['Avg Order Value'].apply(lambda x: f"${x:,.0f}")
    display_profiles['Online %'] = display_profiles['Online %'].apply(lambda x: f"{x*100:.0f}%")
    display_profiles['Rx %'] = display_profiles['Rx %'].apply(lambda x: f"{x*100:.0f}%")
    display_profiles['Size %'] = display_profiles['Size %'].apply(lambda x: f"{x:.1f}%")
    
    st.dataframe(display_profiles[['Segment Name', 'Count', 'Size %', 'Avg Monetary', 
                                   'Avg Frequency', 'Avg Recency', 'Online %', 'Rx %']],
                use_container_width=True, hide_index=True)

# ============================================
# TAB 5: Methodology
# ============================================
with tab5:
    st.header("Segmentation Methodology")
    
    st.markdown("""
    ## Customer Segmentation Approach
    
    ### 1. Feature Engineering
    
    **RFM Features (Core):**
    - Recency: Days since last purchase
    - Frequency: Total number of purchases
    - Monetary: Total customer spend
    
    **Behavioral Features (Extended):**
    - Online purchase ratio
    - Prescription vs sunglasses ratio
    - Average order value
    - Customer tenure
    
    ---
    
    ### 2. Data Preprocessing
    
    ```python
    from sklearn.preprocessing import StandardScaler
    
    # Select features
    features = ['recency_days', 'frequency', 'monetary_value',
                'online_purchase_ratio', 'rx_purchase_ratio', 'avg_order_value']
    
    # Standardize (mean=0, std=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    ```
    
    **Why standardize?**
    - K-means uses Euclidean distance
    - Features on different scales would bias results
    - Recency (days) vs Monetary ($$$) need normalization
    
    ---
    
    ### 3. Optimal Cluster Selection
    
    **Methods:**
    
    | Method | What it measures |
    |--------|------------------|
    | Elbow Method | Within-cluster variance (look for "elbow" in plot) |
    | Silhouette Score | Cluster separation (-1 to 1, higher is better) |
    | Gap Statistic | Compare to null reference distribution |
    | Business Sense | Are segments actionable and distinct? |
    
    ```python
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    
    scores = []
    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        scores.append(silhouette_score(X_scaled, labels))
    
    # Plot scores to find optimal k
    ```
    
    ---
    
    ### 4. K-Means Clustering
    
    ```python
    from sklearn.cluster import KMeans
    
    # Fit model
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Get cluster centers
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    ```
    
    **K-Means Algorithm:**
    1. Initialize k random centroids
    2. Assign each point to nearest centroid
    3. Recalculate centroids as cluster means
    4. Repeat until convergence
    
    ---
    
    ### 5. Cluster Interpretation
    
    **For each cluster, analyze:**
    - Size (% of customers)
    - Average feature values
    - Behavioral patterns
    - Business implications
    
    **Naming convention:**
    - Use descriptive, actionable names
    - "High-Value Loyalists" not "Cluster 3"
    - Should immediately suggest marketing strategy
    
    ---
    
    ### 6. Validation
    
    | Validation Type | Approach |
    |-----------------|----------|
    | **Statistical** | Silhouette score, Davies-Bouldin index |
    | **Stability** | Re-run with different seeds, check consistency |
    | **Business** | Do segments make intuitive sense? Are they actionable? |
    | **Predictive** | Can we predict outcomes (churn, LTV) differently by segment? |
    
    ---
    
    ### 7. Activation & Targeting
    
    **For Lookalike Audiences:**
    1. Export high-value segment customer IDs
    2. Upload to Meta/Google as seed audience
    3. Platform finds similar users to target
    
    **For Personalization:**
    1. Score new customers in real-time
    2. Assign to segment based on nearest centroid
    3. Trigger segment-specific campaigns
    
    ---
    
    ### Alternative Approaches
    
    | Method | When to use |
    |--------|-------------|
    | **Hierarchical** | When you want to explore different granularities |
    | **DBSCAN** | When clusters have irregular shapes |
    | **Gaussian Mixture** | When you need soft/probabilistic assignments |
    | **LDA (Topic Model)** | For text-based behavioral clustering |
    """)
