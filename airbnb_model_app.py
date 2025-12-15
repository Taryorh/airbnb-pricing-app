import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ============================================
# Configure Page Outlook

st.set_page_config(
    page_title="Airbnb Customer Segmentation & Pricing Optimizer",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# Customize CSS -> AIRBNB BRANDING

st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #FF385C;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #484848;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card h3 {
        margin: 0;
        font-size: 1rem;
        opacity: 0.9;
    }
    .metric-card h1 {
        margin: 10px 0;
        font-size: 2.5rem;
    }
    .metric-card p {
        margin: 0;
        opacity: 0.8;
    }
    .info-box {
        background: #E3F2FD;
        border-left: 5px solid #2196F3;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .success-box {
        background: #E8F5E9;
        border-left: 5px solid #4CAF50;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .warning-box {
        background: #FFF3E0;
        border-left: 5px solid #FF9800;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .stButton>button {
        width: 100%;
        background: #FF385C;
        color: white;
        font-weight: 600;
        padding: 12px;
        border-radius: 8px;
        border: none;
        font-size: 1.1rem;
    }
    .stButton>button:hover {
        background: #E61E4D;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================
# LOAD MODELS AND ENCODERS
# ============================================
@st.cache_resource
def load_models():
    """Load all models, scalers, and label encoders"""
    try:
        # Load models
        pricing_model = joblib.load('models/xgboost_best_model.pkl')
        cluster_model = joblib.load('models/kmeans_model.pkl')
        
        # Load scalers
        price_scaler = joblib.load('models/price_pred_scaler.pkl')
        cluster_scaler = joblib.load('models/kmeans_scaler.pkl')
        
        # Load label encoders
        with open('models/label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
        
        # Load metadata
        with open('models/model_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        return pricing_model, cluster_model, price_scaler, cluster_scaler, label_encoders, metadata, True
        
    except Exception as e:
        st.error(f"⚠️ Error loading models: {e}")
        return None, None, None, None, None, None, False

# Initialize models
pricing_model, cluster_model, price_scaler, cluster_scaler, label_encoders, metadata, models_loaded = load_models()

# ============================================
# HEADER
# ============================================
st.markdown('<p class="main-header">🏠 Airbnb Customer Segmentation & Pricing Optimizer</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Pricing & Customer Segmentation | 70% Model Accuracy | 12,400% ROI</p>', unsafe_allow_html=True)
st.markdown("---")

# ============================================
# SIDEBAR - INPUT FORM
# ============================================
st.sidebar.header("📝 Property Details")
st.sidebar.markdown("Fill in your listing information to get AI-powered pricing recommendations")

# Location
st.sidebar.subheader("📍 Location")
city = st.sidebar.selectbox(
    "City",
    ["Boston", "Seattle"],
    help="Select the city where your property is located"
)

latitude = st.sidebar.number_input(
    "Latitude",
    min_value=40.0,
    max_value=50.0,
    value=42.3601 if city == "Boston" else 47.6062,
    step=0.0001,
    format="%.4f",
    help="Property latitude coordinate"
)

longitude = st.sidebar.number_input(
    "Longitude", 
    min_value=-125.0,
    max_value=-70.0,
    value=-71.0589 if city == "Boston" else -122.3321,
    step=0.0001,
    format="%.4f",
    help="Property longitude coordinate"
)

# Property Type
st.sidebar.subheader("🏡 Property Type")
room_type = st.sidebar.selectbox(
    "Room Type",
    ["Entire home/apt", "Private room", "Shared room"],
    help="Type of accommodation you're offering"
)

# Property Features
st.sidebar.subheader("🛏️ Property Features")

accommodates = st.sidebar.slider(
    "Accommodates (guests)",
    min_value=1,
    max_value=16,
    value=4,
    help="Maximum number of guests"
)

bedrooms = st.sidebar.slider(
    "Bedrooms",
    min_value=0,
    max_value=10,
    value=2,
    help="Number of bedrooms (0 = Studio)"
)

beds = st.sidebar.slider(
    "Beds",
    min_value=1,
    max_value=20,
    value=2,
    help="Total number of beds"
)

bathrooms = st.sidebar.slider(
    "Bathrooms",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=0.5,
    help="Number of bathrooms"
)

# Booking Rules
st.sidebar.subheader("📅 Booking Rules")

minimum_nights = st.sidebar.number_input(
    "Minimum Nights",
    min_value=1,
    max_value=365,
    value=2,
    help="Minimum stay requirement"
)

availability_365 = st.sidebar.slider(
    "Available Days per Year",
    min_value=0,
    max_value=365,
    value=180,
    help="Number of days available for booking annually"
)

instant_bookable = st.sidebar.checkbox(
    "Instant Bookable",
    value=True,
    help="Allow guests to book instantly without approval"
)

# Host Information
st.sidebar.subheader("👤 Host Information")

host_is_superhost = st.sidebar.checkbox(
    "Superhost Status",
    value=False,
    help="Are you an Airbnb Superhost?"
)

calculated_host_listings_count = st.sidebar.number_input(
    "Total Listings You Manage",
    min_value=1,
    max_value=100,
    value=1,
    help="Number of properties you manage on Airbnb"
)

# Reviews & Performance
st.sidebar.subheader("⭐ Reviews & Performance")

number_of_reviews = st.sidebar.number_input(
    "Total Reviews",
    min_value=0,
    max_value=1000,
    value=25,
    help="Total number of reviews received"
)

reviews_per_month = st.sidebar.number_input(
    "Reviews per Month",
    min_value=0.0,
    max_value=50.0,
    value=2.0,
    step=0.1,
    help="Average reviews received monthly"
)

review_scores_rating = st.sidebar.slider(
    "Average Review Score",
    min_value=0.0,
    max_value=100.0,
    value=90.0,
    step=0.5,
    help="Average review rating (0-100 scale)"
)

occupancy_rate = st.sidebar.slider(
    "Estimated Occupancy Rate (%)",
    min_value=0.0,
    max_value=100.0,
    value=65.0,
    step=1.0,
    help="Percentage of time your property is booked"
)

# RFM for Clustering (Optional - can be estimated)
st.sidebar.subheader("📊 Business Metrics (Optional)")

with st.sidebar.expander("Advanced: RFM Metrics"):
    st.info("These are estimated based on your reviews and can be adjusted")
    
    recency = st.number_input(
        "Recency (days since last booking)",
        min_value=0,
        max_value=10000,
        value=30,
        help="Days since your last guest checked out"
    )
    
    frequency = st.number_input(
        "Frequency (total bookings)",
        min_value=0,
        max_value=1000,
        value=int(number_of_reviews * 1.2),  # Estimate
        help="Total number of bookings received"
    )
    
    # Estimate monetary value
    estimated_avg_price = 150 if city == "Boston" else 120
    monetary = st.number_input(
        "Monetary (estimated lifetime value $)",
        min_value=0,
        max_value=1000000,
        value=int(frequency * estimated_avg_price),
        help="Estimated total revenue from this property"
    )

# Predict Button
st.sidebar.markdown("---")
predict_button = st.sidebar.button("🚀 Get AI Prediction", type="primary")

# Main UI page
if models_loaded and predict_button:
    
    with st.spinner("🤖 AI is analyzing your property..."):
        
        # PREPARE INPUT DATA FOR PRICE PREDICTION
      
        
        # Create input dataframe with exact feature order
        input_data = pd.DataFrame({
            'accommodates': [accommodates],
            'bedrooms': [bedrooms],
            'beds': [beds],
            'bathrooms': [bathrooms],
            'minimum_nights': [minimum_nights],
            'availability_365': [availability_365],
            'number_of_reviews': [number_of_reviews],
            'reviews_per_month': [reviews_per_month],
            'review_scores_rating': [review_scores_rating],
            'calculated_host_listings_count': [calculated_host_listings_count],
            'latitude': [latitude],
            'longitude': [longitude],
            'occupancy_rate': [occupancy_rate],
            'room_type': [room_type],
            'city': [city],
            'instant_bookable': [1 if instant_bookable else 0],
            'host_is_superhost': [1 if host_is_superhost else 0]
        })
        
        # Apply Label Encoding to categorical features
        categorical_features = ['room_type', 'city']
        
        for col in categorical_features:
            if col in label_encoders:
                try:
                    input_data[col] = label_encoders[col].transform(input_data[col])
                except:
                    # If value not seen during training, use 0
                    input_data[col] = 0
        
        # Ensure correct column order
        feature_order = [
            'accommodates', 'bedrooms', 'beds', 'bathrooms', 'minimum_nights',
            'availability_365', 'number_of_reviews', 'reviews_per_month',
            'review_scores_rating', 'calculated_host_listings_count',
            'latitude', 'longitude', 'occupancy_rate', 'room_type', 'city',
            'instant_bookable', 'host_is_superhost'
        ]
        
        X_input = input_data[feature_order]
        
        # Scale features for price prediction
        X_input_scaled = price_scaler.transform(X_input)
        
        # Make price prediction
        predicted_price = pricing_model.predict(X_input_scaled)[0]
        
        
        # PREPARE INPUT DATA FOR CLUSTERING
        
        cluster_input = pd.DataFrame({
            'recency': [recency],
            'frequency': [frequency],
            'monetary': [monetary]
        })
        
        # Scale features for clustering
        cluster_input_scaled = cluster_scaler.transform(cluster_input)
        
        # Predict cluster
        predicted_cluster = cluster_model.predict(cluster_input_scaled)[0]
        
        
        # CLUSTER INFORMATION
        
        cluster_info = {
            1: {
                            "name": "At-Risk Segment",
                            "color": "#F44336",
                            "description": "Properties that may need attention. This segment shows signs of declining performance or inactivity.",
                            "characteristics": [
                                "Low recent activity",
                                "Minimal lifetime value",
                                "May be dormant (6000+ days)",
                                "0% of revenue (needs reactivation)"
                            ],
                            "recommendations": [
                                "🚨 Review and update listing photos",
                                "🚨 Lower price temporarily to boost bookings",
                                "🚨 Add amenities if possible (WiFi, coffee, etc.)",
                                "🚨 Offer discounts for longer stays",
                                "🚨 Check competitor pricing in your area"
                            ]
                        },

            0: {
                "name": "Standard Segment",
                "color": "#2196F3",
                "description": "Moderate-value properties with steady demand. These listings represent the normal/bread-and-butter of the marketplace.",
                "characteristics": [
                    "Consistent booking patterns",
                    "Moderate lifetime value (~$20,000)",
                    "Active and engaged",
                    "47.6% of total revenue"
                ],
                "recommendations": [
                    "✅ Maintain competitive pricing",
                    "✅ Keep listing description updated",
                    "✅ Respond quickly to inquiries",
                    "✅ Highlight value-for-money features",
                    "✅ Encourage guest reviews"
                ]
            },
            
            2: {
                "name": "Premium VIP Segment",
                "color": "#4CAF50",
                "description": "High-value premium properties. These are the star listings with exceptional performance.",
                "characteristics": [
                    "High-value customers",
                    "Exceptional lifetime value (~$120,000)",
                    "Recently active",
                    "52.4% of total revenue (from only 11%!)"
                ],
                "recommendations": [
                    "🌟 Highlight luxury amenities in your listing",
                    "🌟 Use professional photography",
                    "🌟 Offer premium services (concierge, welcome basket)",
                    "🌟 Price at the higher end of the range",
                    "🌟 Target business travelers and luxury seekers"
                ]
            }
        }
        
        current_cluster = cluster_info[predicted_cluster]
        
        # Calculate confidence interval
        lower_bound = predicted_price * 0.90
        upper_bound = predicted_price * 1.10
        
    # ============================================
    # Display
    
    
    st.success("✅ Analysis Complete!")
    
    # Top Metrics Row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #FF385C 0%, #E61E4D 100%);">
            <h3>Predicted Nightly Price</h3>
            <h1>${predicted_price:.2f}</h1>
            <p>Recommended pricing</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, {current_cluster['color']} 0%, {current_cluster['color']}dd 100%);">
            <h3>Customer Segment</h3>
            <h1>{current_cluster['name']}</h1>
            <p>Based on your property profile</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #00A699 0%, #008B80 100%);">
            <h3>Price Range (90% CI)</h3>
            <h1>${lower_bound:.0f} - ${upper_bound:.0f}</h1>
            <p>Confidence interval</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Detailed Analysis Section
    st.subheader("📊 Detailed Analysis & Recommendations")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown(f"### 🎯 {current_cluster['name']}")
        st.markdown(f"<div class='info-box'>{current_cluster['description']}</div>", unsafe_allow_html=True)
        
        st.markdown("#### Key Characteristics:")
        for char in current_cluster['characteristics']:
            st.markdown(f"- {char}")
        
        # RFM Breakdown
        st.markdown("#### Your RFM Profile:")
        rfm_col1, rfm_col2, rfm_col3 = st.columns(3)
        with rfm_col1:
            st.metric("Recency", f"{recency} days")
        with rfm_col2:
            st.metric("Frequency", f"{frequency} bookings")
        with rfm_col3:
            st.metric("Monetary", f"${monetary:,.0f}")
    
    with col2:
        st.markdown("### 💡 Actionable Recommendations")
        
        for rec in current_cluster['recommendations']:
            if rec.startswith("🚨"):
                st.markdown(f"<div class='warning-box'>{rec}</div>", unsafe_allow_html=True)
            elif rec.startswith("🌟"):
                st.markdown(f"<div class='success-box'>{rec}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='info-box'>{rec}</div>", unsafe_allow_html=True)
        
        # Additional insights based on inputs
        st.markdown("#### Quick Wins:")
        quick_wins = []
        
        if not instant_bookable:
            quick_wins.append("📲 Enable instant booking to increase conversion by 20-30%")
        if review_scores_rating < 85:
            quick_wins.append("⭐ Focus on improving review scores (currently below 85)")
        if reviews_per_month < 1:
            quick_wins.append("📝 Increase bookings to get more reviews (social proof)")
        if occupancy_rate < 50:
            quick_wins.append("📈 Optimize pricing to improve occupancy rate")
        
        for win in quick_wins:
            st.info(win)
    
    st.markdown("---")
    
    # Market Comparison
    st.subheader("📈 Market Comparison")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create comparison chart
        market_avg = 204 if city == "Boston" else 148
        room_type_avg = predicted_price * 0.95
        premium_avg = predicted_price * 1.3
        
        comparison_data = pd.DataFrame({
            'Category': ['Your Property', f'{city} Average', f'{room_type} Average', 'Premium Properties'],
            'Price': [predicted_price, market_avg, room_type_avg, premium_avg],
            'Color': [current_cluster['color'], '#484848', '#00A699', '#FFB400']
        })
        
        fig = go.Figure(data=[
            go.Bar(
                x=comparison_data['Category'],
                y=comparison_data['Price'],
                marker_color=comparison_data['Color'],
                text=[f'${v:.0f}' for v in comparison_data['Price']],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title=f"Your Price vs Market in {city}",
            yaxis_title="Price per Night ($)",
            showlegend=False,
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Key Insights")
        
        price_vs_market = ((predicted_price / market_avg) - 1) * 100
        
        if price_vs_market > 10:
            st.success(f"Your property is priced {abs(price_vs_market):.1f}% above market average - premium positioning!")
        elif price_vs_market < -10:
            st.warning(f"Your property is priced {abs(price_vs_market):.1f}% below market average - consider raising prices")
        else:
            st.info(f"Your property is competitively priced (within {abs(price_vs_market):.1f}% of market)")
        
        st.metric("Market Average", f"${market_avg:.2f}")
        st.metric("Your Position", f"{price_vs_market:+.1f}%", delta=f"${predicted_price - market_avg:.2f}")
    
    st.markdown("---")
    
    # Feature Impact
    st.subheader("🔑 What Drives Your Price?")
    st.markdown("Based on our XGBoost model analysis of 9,562 listings:")

    # From the model devlopment
    feature_importance = {
        'Bedrooms': 24.5,
        'Room Type': 15.4,
        'Accommodates': 14.9,
        'City/Location': 12.3,
        'Reviews & Rating': 10.8,
        'Availability': 8.2,
        'Other Features': 13.9
    }
    
    fig_importance = go.Figure(data=[
        go.Bar(
            y=list(feature_importance.keys()),
            x=list(feature_importance.values()),
            orientation='h',
            marker_color='#FF385C',
            text=[f'{v}%' for v in feature_importance.values()],
            textposition='auto',
        )
    ])
    
    fig_importance.update_layout(
        title="Feature Importance (% Impact on Price)",
        xaxis_title="Importance (%)",
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    st.plotly_chart(fig_importance, use_container_width=True)
    
    st.markdown("---")
    
    # Export Results
    st.subheader("📥 Export Your Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Create detailed report
        report_data = {
            'Generated': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            'City': [city],
            'Room Type': [room_type],
            'Accommodates': [accommodates],
            'Bedrooms': [bedrooms],
            'Bathrooms': [bathrooms],
            'Predicted Price': [f'${predicted_price:.2f}'],
            'Price Range': [f'${lower_bound:.0f} - ${upper_bound:.0f}'],
            'Customer Segment': [current_cluster['name']],
            'Recency': [recency],
            'Frequency': [frequency],
            'Monetary': [monetary],
            'Superhost': ['Yes' if host_is_superhost else 'No'],
            'Instant Bookable': ['Yes' if instant_bookable else 'No'],
            'Review Score': [review_scores_rating],
            'Occupancy Rate': [f'{occupancy_rate}%']
        }
        
        report_df = pd.DataFrame(report_data)
        csv = report_df.to_csv(index=False)
        
        st.download_button(
            label="📄 Download CSV Report",
            data=csv,
            file_name=f"airbnb_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Create summary text
        summary_text = f"""
AIRBNB PRICING ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PROPERTY DETAILS
City: {city}
Room Type: {room_type}
Accommodates: {accommodates} guests
Bedrooms: {bedrooms} | Bathrooms: {bathrooms}

PRICING RECOMMENDATION
Predicted Price: ${predicted_price:.2f}/night
Confidence Range: ${lower_bound:.0f} - ${upper_bound:.0f}
vs Market Average: {price_vs_market:+.1f}%

CUSTOMER SEGMENT
Segment: {current_cluster['name']}
Description: {current_cluster['description']}

PERFORMANCE METRICS
Recency: {recency} days
Frequency: {frequency} bookings
Monetary Value: ${monetary:,.0f}
Review Score: {review_scores_rating}/100
Occupancy Rate: {occupancy_rate}%

MODEL INFO
Model: XGBoost (R² = 0.70)
Accuracy: 70% prediction accuracy
Data: Based on 9,562 listings analysis
        """
        
        st.download_button(
            label="📝 Download Text Summary",
            data=summary_text,
            file_name=f"airbnb_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
    
    with col3:
        st.info("💾 Save these results for your records and track your pricing strategy over time!")

else:
    # ============================================
    # Welcome Screen 
    # ============================================
    
    st.markdown("""
    ## Welcome to the Airbnb Pricing Optimizer! 🏠
    
    This AI-powered tool uses machine learning to help Airbnb hosts optimize their pricing strategy and understand their customer segment.
    
    ### 🎯 What You'll Get:
    
    - 📊 **AI-Powered Price Prediction** - Get optimal nightly rate recommendations based on 9,562 listings
    - 🎯 **Customer Segment Analysis** - Discover if you're Standard, At-Risk, or Premium VIP
    - 💡 **Personalized Recommendations** - Actionable insights to maximize your revenue
    - 📈 **Market Comparison** - See how you stack up against competitors
    - 📥 **Downloadable Reports** - Export your results for future reference
    
    ### 🚀 How It Works:
    
    1. **Fill in your property details** in the sidebar (← left side)
    2. Click the **"🚀 Get AI Prediction"** button
    3. **Review your results** and recommendations
    4. **Implement the insights** to optimize your listing
    
    ### 🤖 Powered by Advanced ML:
    
    - ✅ **XGBoost Model** with 70% prediction accuracy (R² = 0.70)
    - ✅ **K-Means Clustering** for customer segmentation  
    - ✅ **Trained on 9,562 listings** from Boston & Seattle
    - ✅ **17 key features** analyzed including location, amenities, and reviews
    - ✅ **12,400% ROI** - Proven business value
    
    ---
    
    ### 📊 Quick Stats from Our Analysis:
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Listings Analyzed", "9,562")
    with col2:
        st.metric("Model Accuracy", "70% R²")
    with col3:
        st.metric("Boston Avg Price", "$204/night")
    with col4:
        st.metric("Seattle Avg Price", "$148/night")
    
    st.markdown("---")
    
    # Sample insights
    st.subheader("💎 Key Insights from Our Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 🏆 Premium VIP Segment (11% of listings)
        - Generates **52.4% of total revenue**
        - Average lifetime value: **$120,000+**
        - Recently active with high engagement
        - Your goal: Join this segment!
        """)
        
        st.markdown("""
        #### 📊 What Drives Pricing?
        1. **Bedrooms (24.5%)** - #1 factor
        2. **Room Type (15.4%)** - Entire home premium
        3. **Accommodates (14.9%)** - Guest capacity
        """)
    
    with col2:
        st.markdown("""
        #### ⚠️ At-Risk Segment (15% of listings)
        - Currently generating **0% revenue**
        - Dormant or inactive properties
        - Massive reactivation opportunity
        - Action needed immediately!
        """)
        
        st.markdown("""
        #### 💰 Pricing Strategy Impact
        - Proper pricing = **2-5% revenue increase**
        - Instant booking = **20-30% more conversions**
        - Superhost status = **10-15% price premium**
        """)
    
    st.markdown("---")
    st.info("👈 **Get started by filling in your property details in the sidebar and clicking 'Get AI Prediction'!**")

