import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(page_title="YouTube Insights Dashboard 📺", layout="wide", page_icon="🎥")

# Title and introduction
st.title("🎥 YouTube Insights Dashboard")
st.markdown("""
Welcome to your YouTube analytics hub! 🚀 Dive into clear, emoji-packed insights to boost your channel's performance. Explore trends 📈, uncover revenue drivers 💰, and predict earnings with ease! 😊
""")

# Read CSV
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(r"C:\Users\pc\OneDrive\Documents\cip 2025\Data Analyst\youtube\youtube_channel_real_performance_analytics.csv")
        return df
    except FileNotFoundError:
        st.error("🚫 Error: 'youtube_data.csv' not found. Place the file in the same directory.")
        return None

df = load_data()
if df is None:
    st.stop()

# Preprocess data
def preprocess_data(df):
    df['Video Publish Time'] = pd.to_datetime(df['Video Publish Time'], errors='coerce')
    df['Publish Year'] = df['Video Publish Time'].dt.year.fillna(0).astype(int)
    df['Publish Month'] = df['Video Publish Time'].dt.month.fillna(0).astype(int)
    df = pd.get_dummies(df, columns=['Day of Week'], drop_first=True)
    drop_columns = ['ID', 'Video Publish Time']
    df_numeric = df.drop(columns=[col for col in drop_columns if col in df.columns])
    df_numeric = df_numeric.fillna(0)
    return df_numeric

# Sidebar navigation
st.sidebar.header("Navigate Your Insights 🧭")
st.sidebar.markdown("Pick a section to explore your channel's stats! 📊")
page = st.sidebar.selectbox("", ["Channel Snapshot 📸", "Performance Trends 📈", "Revenue Drivers 💸", "Earnings Predictor 🔮"])

# Channel Snapshot
if page == "Channel Snapshot 📸":
    st.header("Channel Snapshot 📸")
    st.markdown("🔍 Get a quick look at your channel's key stats and video performance!")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Videos 🎬", f"{len(df):,}")
    with col2:
        st.metric("Total Views 👀", f"{int(df['Views'].sum()):,}")
    with col3:
        st.metric("Total Revenue 💰", f"${df['Estimated Revenue (USD)'].sum():,.2f}")
    with col4:
        st.metric("Avg. Likes 👍", f"{df['Likes'].mean():,.0f}/Video")
    
    # Sample Data
    st.subheader("Your Videos 🎞️")
    st.dataframe(df[['Video Duration', 'Views', 'Estimated Revenue (USD)', 'Likes', 'Day of Week', 'Video Thumbnail CTR (%)']].head(), use_container_width=True)
    
    # Summary Stats
    st.subheader("Key Stats 📊")
    stats = pd.DataFrame({
        'Metric': ['Avg. Revenue/Video 💵', 'Avg. Views/Video 👀', 'Avg. Watch Time ⏱️', 'Avg. Thumbnail CTR 🎨'],
        'Value': [
            f"${df['Estimated Revenue (USD)'].mean():,.2f}",
            f"{df['Views'].mean():,.0f}",
            f"{df['Watch Time (hours)'].mean():,.2f} hours",
            f"{df['Video Thumbnail CTR (%)'].mean():,.2f}%"
        ]
    })
    st.table(stats)
    st.markdown("💡 **Insight**: High thumbnail CTR and watch time are your channel's superpowers! Focus on these to grow views and revenue. 🚀")

# Performance Trends
elif page == "Performance Trends 📈":
    st.header("Performance Trends 📈")
    st.markdown("🌟 Discover how your videos perform with easy-to-read charts!")
    
    # Revenue Distribution
    st.subheader("How Much Are Your Videos Earning? 💰")
    fig = px.histogram(df, x='Estimated Revenue (USD)', nbins=25, 
                       title="Revenue Distribution",
                       color_discrete_sequence=['#ff6b6b'], marginal="box")
    fig.update_layout(xaxis_title="Revenue (USD)", yaxis_title="Number of Videos", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("💡 **Insight**: Most videos earn in a specific range. Check outliers to see what makes top earners shine! ✨")
    
    # Views vs Revenue
    st.subheader("Do Views Mean More Money? 👀💵")
    fig = px.scatter(df, x='Views', y='Estimated Revenue (USD)', 
                     size='Watch Time (hours)', color='Video Thumbnail CTR (%)',
                     hover_data=['Video Duration', 'Likes'],
                     title="Views vs Revenue (Bubble Size = Watch Time)")
    fig.update_layout(xaxis_title="Views", yaxis_title="Revenue (USD)")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("💡 **Insight**: Bigger bubbles (more watch time) and brighter colors (higher CTR) often mean more revenue. Optimize these! 🎯")
    
    # Revenue by Day of Week
    st.subheader("Best Days to Post 📅")
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    avg_revenue = df.groupby('Day of Week')['Estimated Revenue (USD)'].mean().reindex(day_order).reset_index()
    fig = px.bar(avg_revenue, x='Day of Week', y='Estimated Revenue (USD)', 
                 title="Average Revenue by Posting Day",
                 color_discrete_sequence=['#00cc96'])
    fig.update_layout(xaxis_title="Day of Week", yaxis_title="Average Revenue (USD)")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("💡 **Insight**: Some days bring in more cash! Try posting on high-revenue days to boost earnings. 🤑")
    
    # Engagement Breakdown
    st.subheader("How Engaged Are Your Fans? 😍")
    engagement_data = pd.DataFrame({
        'Metric': ['Likes 👍', 'Comments 💬', 'Shares 🔗', 'New Subscribers 🔔'],
        'Total': [df['Likes'].sum(), df['New Comments'].sum(), df['Shares'].sum(), df['New Subscribers'].sum()]
    })
    fig = px.pie(engagement_data, values='Total', names='Metric', 
                 title="Audience Engagement Breakdown",
                 color_discrete_sequence=px.colors.qualitative.Pastel, hole=0.4)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("💡 **Insight**: Strong likes and comments drive growth. Encourage fans to engage more! 🚀")

# Revenue Drivers
elif page == "Revenue Drivers 💸":
    st.header("Revenue Drivers 💸")
    st.markdown("🔎 Find out what makes your channel earn more!")
    
    df_numeric = preprocess_data(df)
    X = df_numeric.drop(columns=['Estimated Revenue (USD)'])
    y = df_numeric['Estimated Revenue (USD)']
    
    # Train Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Feature Importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    st.subheader("Top Money Makers 💰")
    fig = px.bar(feature_importance.head(12), x='Importance', y='Feature', 
                 orientation='h', title="Top 12 Factors Driving Revenue",
                 color='Importance', color_continuous_scale='Reds')
    fig.update_layout(yaxis={'tickmode': 'linear'}, xaxis_title="Importance Score", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("💡 **Insight**: Focus on top drivers like Watch Time and Ad Impressions to skyrocket your revenue! 🌟")
    
    # Feature Correlation
    st.subheader("How Metrics Impact Revenue 📊")
    key_features = ['Views', 'Watch Time (hours)', 'Ad Impressions', 'Video Thumbnail CTR (%)']
    corr_data = df[key_features + ['Estimated Revenue (USD)']].corr()['Estimated Revenue (USD)'].iloc[:-1]
    corr_df = pd.DataFrame({'Feature': corr_data.index, 'Correlation': corr_data.values})
    fig = px.bar(corr_df, x='Correlation', y='Feature', orientation='h',
                 title="Metric Impact on Revenue",
                 color='Correlation', color_continuous_scale='Blues')
    fig.update_layout(xaxis_title="Correlation with Revenue", yaxis_title="", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("💡 **Insight**: Metrics with strong positive correlation (like Views) are key to higher earnings. Boost these! 💪")

# Earnings Predictor
elif page == "Earnings Predictor 🔮":
    st.header("Earnings Predictor 🔮")
    st.markdown("🎯 Input video details to predict how much you could earn!")
    
    df_numeric = preprocess_data(df)
    X = df_numeric.drop(columns=['Estimated Revenue (USD)'])
    y = df_numeric['Estimated Revenue (USD)']
    
    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    
    # Model Performance
    y_pred = rf.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    st.subheader("Prediction Accuracy 📏")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean Squared Error", f"{mse:.2f}", help="Lower is better!")
    with col2:
        st.metric("Mean Absolute Error", f"${mae:.2f}", help="Average prediction error in USD")
    with col3:
        st.metric("Accuracy (R²)", f"{r2:.2%}", help="Higher means better predictions!")
    st.markdown("💡 **Insight**: The model’s accuracy shows how reliable its predictions are. Use it to plan your next video! 🚀")
    
    # Actual vs Predicted
    st.subheader("How Good Are Our Predictions? 🎯")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', 
                            name='Predictions', marker=dict(size=10, color='#ff6b6b')))
    fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], 
                            y=[y_test.min(), y_test.max()], 
                            mode='lines', name='Perfect Prediction', line=dict(dash='dash', color='#2c3e50')))
    fig.update_layout(title="Actual vs Predicted Revenue",
                      xaxis_title="Actual Revenue (USD)", yaxis_title="Predicted Revenue (USD)")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("💡 **Insight**: Dots close to the dashed line mean accurate predictions. Trust the model for reliable forecasts! ✅")
    
    # Interactive Prediction
    st.subheader("Predict Your Earnings 💵")
    st.markdown("🎚️ Adjust sliders to see how video stats affect revenue!")
    
    key_features = [
        'Views', 'Watch Time (hours)', 'Ad Impressions', 'Video Thumbnail CTR (%)',
        'Average View Percentage (%)', 'Likes', 'New Subscribers', 'Monetized Playbacks (Estimate)'
    ]
    
    input_data = {}
    st.markdown("### Video Stats 🎬")
    col1, col2 = st.columns(2)
    for i, feature in enumerate(key_features):
        with col1 if i % 2 == 0 else col2:
            min_val = float(df[feature].min())
            max_val = float(df[feature].max())
            mean_val = float(df[feature].mean())
            input_data[feature] = st.slider(f"{feature} {'👀' if 'Views' in feature else '⏱️' if 'Watch Time' in feature else '💸' if 'Ad Impressions' in feature else '🎨' if 'Thumbnail' in feature else '📈' if 'Percentage' in feature else '👍' if 'Likes' in feature else '🔔' if 'Subscribers' in feature else '🎥'}", 
                                           min_val, max_val, mean_val, step=(max_val-min_val)/100,
                                           help=f"Range: {min_val:.2f} to {max_val:.2f}")
    
    # Prepare input
    input_df = pd.DataFrame([input_data])
    for col in X.columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[X.columns]
    input_scaled = scaler.transform(input_df)
    
    # Predict
    prediction = rf.predict(input_scaled)[0]
    
    st.markdown(f"### Predicted Revenue: <span style='color:#ff6b6b; font-size:1.5em;'>${prediction:.2f} USD</span> 💰", unsafe_allow_html=True)
    st.markdown("💡 **Insight**: Boost Views, Watch Time, or Thumbnail CTR to increase earnings. Play with sliders to find the sweet spot! 🌟")
    st.markdown("🔮 **Tip**: Use this tool to plan your next video and maximize revenue potential! 🚀")