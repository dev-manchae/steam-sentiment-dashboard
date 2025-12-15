import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import os
import re

# ==========================================
# 0. PAGE CONFIG & SETUP
# ==========================================
st.set_page_config(page_title="Steam Sentiment Intelligence", page_icon="🎮", layout="wide")

# PATHS
DATA_FILE = "https://huggingface.co/manchae86/steam-review-roberta/resolve/main/steam_reviews.csv"
BENCHMARK_FILE = "./data/benchmark.csv"

# ==========================================
# 1. CUSTOM CSS (BACKGROUND & UI)
# ==========================================
st.markdown("""
<style>
    /* MAIN BACKGROUND: Steam Blue -> Deep Charcoal Gradient */
    .stApp {
        background: linear-gradient(135deg, #1b2838 0%, #171a21 100%);
        background-attachment: fixed;
        color: white;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a1c24;
        border-right: 1px solid #2d303e;
    }
    
    /* Metric Cards */
    .metric-card {
        background-color: #262730;
        border: 1px solid #3b3c46;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        color: #66c0f4; /* Steam Blue */
    }

    /* Tabs as Buttons */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 55px;
        white-space: pre-wrap;
        background-color: #262730;
        border-radius: 8px 8px 0px 0px; 
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        padding-left: 20px;
        padding-right: 20px;
        border: 1px solid #4c4c4c;
        color: #b0b3b8;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #31333F;
        border-color: #66c0f4;
        color: #66c0f4;
    }
    .stTabs [aria-selected="true"] {
        background-color: #66c0f4;
        color: white;
        border: none;
    }
    
    /* Result Card in Live AI Lab */
    .result-card {
        background-color: #262730;
        border: 1px solid #4c4c4c;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }

    /* Headers */
    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: white;
        margin-bottom: 0px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    .sub-header {
        font-size: 18px;
        color: #b0b3b8;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA LOADING
# ==========================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_FILE)
        
        if 'game_name' in df.columns:
            df.rename(columns={'game_name': 'app_name'}, inplace=True)
        elif 'app_name' not in df.columns:
            df['app_name'] = "Unknown Game"
            
        if 'timestamp' in df.columns:
            df['date'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        return df
    except Exception as e:
        return None

@st.cache_data
def load_benchmark():
    if not os.path.exists(BENCHMARK_FILE):
        return None
    return pd.read_csv(BENCHMARK_FILE)

# --- CLOUD MODEL LOADER ---
@st.cache_resource
def load_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        model = AutoModelForSequenceClassification.from_pretrained("manchae86/steam-review-roberta")
        return tokenizer, model
    except Exception as e:
        return None, None

df = load_data()

# ==========================================
# 3. SIDEBAR CONTROL PANEL
# ==========================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/83/Steam_icon_logo.svg/2048px-Steam_icon_logo.svg.png", width=50)
    st.markdown("## 🎛️ Control Panel")
    
    if df is not None:
        game_stats = df.groupby('app_name')['sentiment'].apply(lambda x: (x == 2).mean()).reset_index(name='sat_rate')
        sorted_games = game_stats.sort_values('sat_rate', ascending=False)['app_name'].tolist()
        game_list = ["All Games"] + sorted_games
        
        selected_game = st.selectbox("Select Game context:", game_list)
        
        if selected_game != "All Games":
            df_filtered = df[df['app_name'] == selected_game]
        else:
            df_filtered = df
            
        total_reviews = len(df_filtered)
        if total_reviews > 0:
            pos_reviews = len(df_filtered[df_filtered['sentiment'] == 2])
            satisfaction_rate = (pos_reviews / total_reviews) * 100
        else:
            satisfaction_rate = 0.0

        st.divider()

        st.markdown(f"""
        <div class="metric-card">
            <div style="color: #b0b3b8;">Total Reviews Analyzed</div>
            <div class="metric-value">{total_reviews:,}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <div style="color: #b0b3b8;">Satisfaction Rate ❔</div>
            <div class="metric-value">{satisfaction_rate:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("ℹ️ Running RoBERTa v4.0 Model")
    else:
        st.error("Data not found. Please ensure URL is correct.")
        df_filtered = pd.DataFrame()
        selected_game = "Unknown"

# ==========================================
# 4. MAIN PAGE
# ==========================================

st.markdown('<div class="main-title">🎮 Steam Sentiment Intelligence</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">Deep learning analysis for <b style="color: #66c0f4;">{selected_game}</b> using RoBERTa transformers.</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["Analytics Dashboard", "Topic Clouds", "Live AI Lab", "Model Benchmarks"])

COLOR_MAP = {
    "Satisfied": "#4DB6AC",
    "Neutral": "#FFB74D",
    "Dissatisfied": "#FF5252"
}

# ==========================================
# 5. TAB CONTENT
# ==========================================

# --- TAB 1: DASHBOARD ---
with tab1:
    if not df_filtered.empty:
        df_filtered['Sentiment Label'] = df_filtered['sentiment'].map({0: 'Dissatisfied', 1: 'Neutral', 2: 'Satisfied'})
        
        c1, c2 = st.columns([1, 2])
        
        with c1:
            st.markdown("### Sentiment Distribution")
            pie_data = df_filtered['Sentiment Label'].value_counts().reset_index()
            pie_data.columns = ['Sentiment', 'Count']
            
            fig_pie = px.pie(
                pie_data, 
                names='Sentiment', values='Count', color='Sentiment',
                color_discrete_map=COLOR_MAP, hole=0.6
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            
            fig_pie.update_layout(
                showlegend=False, 
                margin=dict(t=0, b=0, l=0, r=0), 
                paper_bgcolor="rgba(0,0,0,0)", 
                plot_bgcolor="rgba(0,0,0,0)"
            )
            
            display_name = selected_game.replace("_", " ")
            if len(display_name) > 15: display_name = display_name.replace(" ", "<br>")
            fig_pie.add_annotation(text=f"<b>{display_name}</b>", showarrow=False, font_size=18, font_color="white")
            
            # HIDE MODEBAR HERE
            st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})

        with c2:
            st.markdown("### Sentiment Trends Over Time")
            if 'date' in df_filtered.columns and df_filtered['date'].notna().any():
                timeline_data = df_filtered.groupby([pd.Grouper(key='date', freq='M'), 'Sentiment Label']).size().reset_index(name='Count')
                fig_area = px.area(
                    timeline_data, x='date', y='Count', color='Sentiment Label',
                    color_discrete_map=COLOR_MAP
                )
                fig_area.update_layout(
                    xaxis_title="", yaxis_title="Number of Reviews", margin=dict(t=0, b=0, l=0, r=0),
                    paper_bgcolor="rgba(0,0,0,0)", 
                    plot_bgcolor="rgba(0,0,0,0)", 
                    font=dict(color="white")
                )
                # HIDE MODEBAR HERE
                st.plotly_chart(fig_area, use_container_width=True, config={'displayModeBar': False})
            else:
                st.warning("⚠️ No timestamp data found to build timeline.")

        st.divider()

        if selected_game == "All Games":
            st.markdown("### 🏆 Leaderboard: Most Loved Games")
            lb_data = df.groupby(['app_name', 'sentiment']).size().reset_index(name='count')
            lb_total = df.groupby('app_name').size().reset_index(name='total')
            lb_data = lb_data.merge(lb_total, on='app_name')
            lb_data['percentage'] = lb_data['count'] / lb_data['total']
            lb_data['Sentiment Label'] = lb_data['sentiment'].map({0: 'Dissatisfied', 1: 'Neutral', 2: 'Satisfied'})

            rankings = lb_data[lb_data['sentiment'] == 2].sort_values('percentage', ascending=True)
            sorted_games_order = rankings['app_name'].tolist()

            fig_bar = px.bar(
                lb_data, y='app_name', x='percentage', color='Sentiment Label',
                orientation='h', color_discrete_map=COLOR_MAP, text_auto='.0%',
                category_orders={'app_name': sorted_games_order}
            )
            fig_bar.update_layout(
                barmode='stack', xaxis_tickformat='.0%', yaxis_title="", xaxis_title="Percentage of Reviews",
                margin=dict(t=0, b=0, l=0, r=0), 
                paper_bgcolor="rgba(0,0,0,0)", 
                plot_bgcolor="rgba(0,0,0,0)", 
                font=dict(color="white")
            )
            # HIDE MODEBAR HERE
            st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})
            st.divider()

        st.markdown("### 📏 Review Length Analysis")
        df_filtered['length'] = df_filtered['clean_text'].astype(str).apply(len)
        fig_box = px.box(
            df_filtered, x='Sentiment Label', y='length', color='Sentiment Label',
            color_discrete_map=COLOR_MAP, title="Do angry gamers write longer reviews?"
        )
        fig_box.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)", 
            font=dict(color="white")
        )
        # HIDE MODEBAR HERE
        st.plotly_chart(fig_box, use_container_width=True, config={'displayModeBar': False})
    else:
        st.warning("No data available.")

# --- TAB 2: TOPIC CLOUDS ---
with tab2:
    st.subheader(f"☁️ What are players saying about {selected_game}?")
    if not df_filtered.empty:
        c1, c2 = st.columns([1, 3])
        with c1:
            st.markdown("### Cloud Settings")
            st.write("Sentiment focus:")
            wc_sentiment = st.radio("Select Sentiment", ["Satisfied", "Dissatisfied"], label_visibility="collapsed")
            st.info("💡 **Pro Tip:** We automatically filter out common words like 'game', 'play', and 'steam'.")
            
            if wc_sentiment == "Dissatisfied":
                sent_id = 0; cloud_colormap = 'magma'
            else:
                sent_id = 2; cloud_colormap = 'viridis'

        with c2:
            subset = df_filtered[df_filtered['sentiment'] == sent_id]
            if not subset.empty:
                text_corpus = " ".join(subset['clean_text'].astype(str).tolist())
                text_corpus = re.sub(r'\bd\d+\b', '', text_corpus)
                
                wc = WordCloud(width=800, height=500, background_color='white', colormap=cloud_colormap, collocations=False).generate(text_corpus)
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.imshow(wc, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig)
            else:
                st.warning(f"No {wc_sentiment} reviews found.")

# --- TAB 3: LIVE AI LAB ---
with tab3:
    st.subheader("🤖 Real-time Sentiment Detection")
    st.write("Test the model with your own custom text.")
    
    tokenizer, model = load_model()
    
    if model:
        col1, col2 = st.columns([2, 1], gap="large")
        with col1:
            text = st.text_area("Enter a review:", height=200, placeholder="Example: The combat is peak but the optimization is trash.")
            analyze = st.button("Analyze Sentiment", type="primary", use_container_width=True)
        
        with col2:
            if analyze and text:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
                with torch.no_grad():
                    logits = model(**inputs).logits
                probs = F.softmax(logits, dim=-1)
                score = torch.argmax(probs, dim=-1).item()
                confidence = probs[0][score].item()
                
                labels = {0: ("Dissatisfied", "🔴"), 1: ("Neutral", "⚪"), 2: ("Satisfied", "🟢")}
                label_txt, icon = labels[score]
                
                st.markdown(f"""
                <div class="result-card">
                    <h3 style="color:#b0b3b8; margin:0;">Prediction</h3>
                    <h1 style="font-size: 42px; margin: 10px 0;">{icon}</h1>
                    <h2 style="color: white;">{label_txt}</h2>
                </div>
                """, unsafe_allow_html=True)
                
                st.write(""); st.write("Confidence Score:")
                st.progress(confidence)
                st.caption(f"AI Confidence: **{confidence:.1%}**")
            else:
                st.info("👈 Enter text and click **Analyze** to see the AI prediction here.")
    else:
        st.error("⚠️ Model could not be loaded. Please check Hugging Face connection.")

# --- TAB 4: BENCHMARKS ---
with tab4:
    st.subheader("🏆 Model Performance Leaderboard")
    df_bench = load_benchmark()
    if df_bench is not None:
        fig = px.bar(
            df_bench.sort_values("Accuracy", ascending=True), 
            x="Accuracy", y="Model", orientation='h', text_auto='.2%',
            color="Accuracy", color_continuous_scale="Viridis"
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)", 
            font=dict(color="white")
        )
        # HIDE MODEBAR HERE
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.dataframe(df_bench)