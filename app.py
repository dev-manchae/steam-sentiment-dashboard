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
# UPDATED: Pull benchmark directly from Hugging Face so it is always 96% accurate
BENCHMARK_FILE = "https://huggingface.co/manchae86/steam-review-roberta/resolve/main/benchmark.csv"

# ==========================================
# ISSUE TAXONOMY FOR DEVELOPER INSIGHTS
# ==========================================
ISSUE_TAXONOMY = {
    "🖥️ Technical": {
        "keywords": ["lag", "crash", "bug", "fps", "optimization", "server", "freeze", "loading", 
                     "performance", "stuttering", "glitch", "error", "disconnect", "memory", "frame"],
        "color": "#FF6B6B",
        "recommendations": {
            "high": "Consider infrastructure scaling, code profiling, or CDN optimization",
            "medium": "Monitor error logs and prioritize critical fixes in next patch",
            "low": "Add to backlog for future optimization sprint"
        }
    },
    "🎮 Gameplay": {
        "keywords": ["balance", "difficulty", "controls", "mechanics", "combat", "boring", 
                     "repetitive", "grind", "unfair", "broken", "nerf", "buff", "skill", "enemy"],
        "color": "#4ECDC4",
        "recommendations": {
            "high": "Conduct gameplay testing sessions and adjust difficulty curves",
            "medium": "Review player progression data and rebalance mechanics",
            "low": "Gather more feedback before making changes"
        }
    },
    "📖 Content": {
        "keywords": ["story", "dlc", "mission", "quest", "content", "short", "ending", 
                     "character", "update", "more", "expansion", "level", "world", "lore"],
        "color": "#45B7D1",
        "recommendations": {
            "high": "Prioritize content expansion in roadmap - strong revenue potential",
            "medium": "Plan seasonal content updates to maintain engagement",
            "low": "Consider community-driven content or modding support"
        }
    },
    "💰 Monetization": {
        "keywords": ["price", "expensive", "cheap", "worth", "money", "microtransaction", 
                     "p2w", "pay", "cost", "value", "purchase", "sale", "discount"],
        "color": "#FFD93D",
        "recommendations": {
            "high": "Review pricing strategy - consider regional pricing or bundles",
            "medium": "Improve value perception through better communication",
            "low": "Monitor competitor pricing and adjust accordingly"
        }
    }
}

# ==========================================
# DEVELOPER INSIGHTS HELPER FUNCTIONS
# ==========================================
def extract_issues(df, sentiment_filter=None):
    """Extract and categorize issues from reviews based on ISSUE_TAXONOMY keywords."""
    if sentiment_filter is not None:
        subset = df[df['sentiment'] == sentiment_filter]
    else:
        subset = df[df['sentiment'].isin([0, 1])]  # Negative and neutral reviews
    
    issue_counts = {}
    issue_reviews = {}
    
    for category, data in ISSUE_TAXONOMY.items():
        keywords = data["keywords"]
        count = 0
        matching_reviews = []
        
        for _, row in subset.iterrows():
            text = str(row.get('clean_text', '')).lower()
            if any(kw in text for kw in keywords):
                count += 1
                matching_reviews.append(text[:200])  # Store preview
        
        issue_counts[category] = count
        issue_reviews[category] = matching_reviews[:5]  # Top 5 examples
    
    return issue_counts, issue_reviews

def generate_recommendations(issue_counts, total_reviews):
    """Generate AI-powered recommendations based on issue patterns."""
    recommendations = {"high": [], "medium": [], "opportunities": []}
    
    for category, count in issue_counts.items():
        if total_reviews == 0:
            continue
        percentage = (count / total_reviews) * 100
        data = ISSUE_TAXONOMY[category]
        
        if percentage >= 15:
            recommendations["high"].append({
                "category": category,
                "percentage": percentage,
                "count": count,
                "action": data["recommendations"]["high"],
                "color": data["color"]
            })
        elif percentage >= 5:
            recommendations["medium"].append({
                "category": category,
                "percentage": percentage,
                "count": count,
                "action": data["recommendations"]["medium"],
                "color": data["color"]
            })
        elif percentage >= 2:
            recommendations["opportunities"].append({
                "category": category,
                "percentage": percentage,
                "count": count,
                "action": data["recommendations"]["low"],
                "color": data["color"]
            })
    
    return recommendations

def calculate_priority_matrix(issue_counts, df):
    """Calculate data for Priority Matrix (Impact vs Frequency)."""
    matrix_data = []
    
    for category, count in issue_counts.items():
        data = ISSUE_TAXONOMY[category]
        # Frequency = count of mentions
        frequency = count
        
        # Impact = proportion of negative sentiment in reviews mentioning this issue
        keywords = data["keywords"]
        subset = df[df['clean_text'].str.lower().str.contains('|'.join(keywords), na=False)]
        if len(subset) > 0:
            neg_ratio = (subset['sentiment'] == 0).mean()
            impact = neg_ratio * 100  # Scale to 0-100
        else:
            impact = 0
        
        matrix_data.append({
            "Category": category,
            "Frequency": frequency,
            "Impact": impact,
            "Color": data["color"]
        })
    
    return pd.DataFrame(matrix_data)

def calculate_benchmark(df_game, df_all):
    """Calculate how a game compares to the overall average for each issue category."""
    game_counts, _ = extract_issues(df_game)
    all_counts, _ = extract_issues(df_all)
    
    benchmark_data = []
    total_game = len(df_game)
    total_all = len(df_all)
    
    for category in ISSUE_TAXONOMY.keys():
        game_pct = (game_counts[category] / total_game * 100) if total_game > 0 else 0
        all_pct = (all_counts[category] / total_all * 100) if total_all > 0 else 0
        diff = game_pct - all_pct
        
        benchmark_data.append({
            "Category": category,
            "This Game": game_pct,
            "All Games Avg": all_pct,
            "Difference": diff,
            "Status": "🟢 Better" if diff < -2 else ("🔴 Worse" if diff > 2 else "⚪ Similar"),
            "Color": ISSUE_TAXONOMY[category]["color"]
        })
    
    return pd.DataFrame(benchmark_data)

def calculate_issue_trends(df, category):
    """Calculate issue mentions over time for trend analysis."""
    if 'date' not in df.columns or df['date'].isna().all():
        return None
    
    keywords = ISSUE_TAXONOMY[category]["keywords"]
    df_copy = df.copy()
    df_copy['has_issue'] = df_copy['clean_text'].str.lower().str.contains('|'.join(keywords), na=False)
    
    df_copy['month'] = df_copy['date'].dt.to_period('M')
    monthly = df_copy.groupby('month').agg({
        'has_issue': 'sum',
        'clean_text': 'count'
    }).reset_index()
    monthly.columns = ['Month', 'Issue Count', 'Total Reviews']
    monthly['Issue Rate'] = (monthly['Issue Count'] / monthly['Total Reviews'] * 100).round(1)
    monthly['Month'] = monthly['Month'].astype(str)
    
    return monthly

def get_highlighted_quotes(df, category, n=5):
    """Get the most representative quotes for a category with sentiment context."""
    keywords = ISSUE_TAXONOMY[category]["keywords"]
    mask = df['clean_text'].str.lower().str.contains('|'.join(keywords), na=False)
    subset = df[mask].copy()
    
    if subset.empty:
        return []
    
    subset = subset.sort_values('sentiment', ascending=True)
    quotes = []
    for _, row in subset.head(n).iterrows():
        text = str(row.get('clean_text', ''))[:250]
        sentiment = row.get('sentiment', 1)
        sentiment_label = {0: "😡 Dissatisfied", 1: "😐 Neutral", 2: "😊 Satisfied"}.get(sentiment, "Unknown")
        keyword_matches = sum(1 for kw in keywords if kw in text.lower())
        quotes.append({
            "text": text,
            "sentiment": sentiment_label,
            "relevance": keyword_matches,
            "color": "#FF5252" if sentiment == 0 else ("#FFB74D" if sentiment == 1 else "#4DB6AC")
        })
    return quotes

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
    
    /* Developer Insights Cards */
    .insight-card {
        background: linear-gradient(135deg, #262730 0%, #1e1f26 100%);
        border: 1px solid #3b3c46;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .insight-header {
        font-size: 16px;
        color: #b0b3b8;
        margin-bottom: 8px;
    }
    .insight-value {
        font-size: 28px;
        font-weight: 700;
    }
    .recommendation-card {
        background-color: #1e1f26;
        border-left: 4px solid;
        padding: 15px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 10px;
    }
    .priority-high { border-color: #FF6B6B; }
    .priority-medium { border-color: #FFD93D; }
    .priority-low { border-color: #4ECDC4; }
    
    /* Loading Animations */
    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .skeleton {
        background: linear-gradient(90deg, #262730 25%, #3b3c46 50%, #262730 75%);
        background-size: 200% 100%;
        animation: shimmer 1.5s infinite;
        border-radius: 8px;
    }
    
    .skeleton-text {
        height: 16px;
        margin-bottom: 8px;
        width: 80%;
    }
    
    .skeleton-title {
        height: 32px;
        margin-bottom: 16px;
        width: 60%;
    }
    
    .skeleton-chart {
        height: 200px;
        width: 100%;
    }
    
    .skeleton-card {
        background: linear-gradient(90deg, #262730 25%, #3b3c46 50%, #262730 75%);
        background-size: 200% 100%;
        animation: shimmer 1.5s infinite;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 15px;
        height: 120px;
    }
    
    .loading-pulse {
        animation: pulse 1.5s ease-in-out infinite;
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease-out forwards;
    }
    
    /* Spinner */
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid #3b3c46;
        border-radius: 50%;
        border-top-color: #66c0f4;
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Loading overlay */
    .loading-overlay {
        position: relative;
    }
    
    .loading-overlay::after {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(23, 26, 33, 0.8);
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 12px;
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
    try:
        return pd.read_csv(BENCHMARK_FILE)
    except Exception:
        return None

# --- CLOUD MODEL LOADER (UPDATED FOR V3) ---
# Added TTL to ensure it refreshes if you upload a new model
@st.cache_resource(ttl="1h")
def load_model():
    try:
        # Load BOTH tokenizer and model from your repo to ensure compatibility
        tokenizer = AutoTokenizer.from_pretrained("manchae86/steam-review-roberta")
        model = AutoModelForSequenceClassification.from_pretrained("manchae86/steam-review-roberta")
        return tokenizer, model
    except Exception as e:
        return None, None

# --- SKELETON LOADER HELPERS ---
def show_skeleton_card(height=120):
    """Display a skeleton loading card."""
    st.markdown(f'<div class="skeleton-card" style="height: {height}px;"></div>', unsafe_allow_html=True)

def show_skeleton_text(lines=3, width="80%"):
    """Display skeleton text lines."""
    for i in range(lines):
        w = f"{80 - i*10}%" if width == "80%" else width
        st.markdown(f'<div class="skeleton skeleton-text" style="width: {w};"></div>', unsafe_allow_html=True)

def show_skeleton_chart(height=200):
    """Display a skeleton chart placeholder."""
    st.markdown(f'<div class="skeleton skeleton-chart" style="height: {height}px;"></div>', unsafe_allow_html=True)

def show_loading_spinner(text="Loading..."):
    """Display a custom loading spinner with text."""
    st.markdown(f'''
    <div style="display: flex; align-items: center; gap: 10px; padding: 20px;">
        <div class="loading-spinner"></div>
        <span style="color: #b0b3b8;">{text}</span>
    </div>
    ''', unsafe_allow_html=True)

# Load data with loading state
with st.spinner("🔄 Loading Steam reviews data..."):
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
        
        st.info("ℹ️ Running RoBERTa v3.0 (96% Acc)")
    else:
        st.error("Data not found. Please ensure URL is correct.")
        df_filtered = pd.DataFrame()
        selected_game = "Unknown"

# ==========================================
# 4. MAIN PAGE
# ==========================================

st.markdown('<div class="main-title">🎮 Steam Sentiment Intelligence</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">Deep learning analysis for <b style="color: #66c0f4;">{selected_game}</b> using RoBERTa transformers.</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📊 Dashboard", 
    "☁️ Topic Clouds", 
    "🔍 Dev Insights", 
    "🤖 AI Lab", 
    "🏆 Benchmarks",
    "⚔️ Compare Games",
    "🔮 Predictor",
    "📰 Explorer"
])

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
                margin=dict(t=30, b=0, l=0, r=0),
                paper_bgcolor="rgba(0,0,0,0)", 
                plot_bgcolor="rgba(0,0,0,0)",
                modebar=dict(bgcolor='rgba(0,0,0,0)', color='white') 
            )
            
            display_name = selected_game.replace("_", " ")
            if len(display_name) > 15: display_name = display_name.replace(" ", "<br>")
            fig_pie.add_annotation(text=f"<b>{display_name}</b>", showarrow=False, font_size=18, font_color="white")
            
            st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            st.markdown("### Sentiment Trends Over Time")
            if 'date' in df_filtered.columns and df_filtered['date'].notna().any():
                timeline_data = df_filtered.groupby([pd.Grouper(key='date', freq='M'), 'Sentiment Label']).size().reset_index(name='Count')
                fig_area = px.area(
                    timeline_data, x='date', y='Count', color='Sentiment Label',
                    color_discrete_map=COLOR_MAP
                )
                fig_area.update_layout(
                    xaxis_title="", yaxis_title="Number of Reviews", 
                    margin=dict(t=30, b=0, l=0, r=0),
                    paper_bgcolor="rgba(0,0,0,0)", 
                    plot_bgcolor="rgba(0,0,0,0)", 
                    font=dict(color="white"),
                    modebar=dict(bgcolor='rgba(0,0,0,0)', color='white')
                )
                st.plotly_chart(fig_area, use_container_width=True)
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
                margin=dict(t=30, b=0, l=0, r=0),
                paper_bgcolor="rgba(0,0,0,0)", 
                plot_bgcolor="rgba(0,0,0,0)", 
                font=dict(color="white"),
                modebar=dict(bgcolor='rgba(0,0,0,0)', color='white')
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            st.divider()

        st.markdown("### 📏 Review Length Analysis")
        df_filtered['length'] = df_filtered['clean_text'].astype(str).apply(len)
        fig_box = px.box(
            df_filtered, x='Sentiment Label', y='length', color='Sentiment Label',
            color_discrete_map=COLOR_MAP, title="Do angry gamers write longer reviews?"
        )
        fig_box.update_layout(
            margin=dict(t=50, b=0, l=0, r=0),
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)", 
            font=dict(color="white"),
            modebar=dict(bgcolor='rgba(0,0,0,0)', color='white')
        )
        st.plotly_chart(fig_box, use_container_width=True)
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

# --- TAB 3: DEVELOPER INSIGHTS ---
with tab3:
    st.subheader("🔍 Developer Intelligence Report")
    st.markdown(f"*Actionable insights from player feedback for **{selected_game}***")
    
    if not df_filtered.empty:
        # Extract issues from reviews
        issue_counts, issue_reviews = extract_issues(df_filtered)
        total_issues = sum(issue_counts.values())
        
        # --- EXECUTIVE SUMMARY ---
        st.markdown("### 📊 Executive Summary")
        cols = st.columns(4)
        
        for idx, (category, count) in enumerate(issue_counts.items()):
            with cols[idx]:
                percentage = (count / len(df_filtered)) * 100 if len(df_filtered) > 0 else 0
                color = ISSUE_TAXONOMY[category]["color"]
                st.markdown(f"""
                <div class="insight-card fade-in" style="border-top: 3px solid {color}; animation-delay: {idx * 0.1}s;">
                    <div class="insight-header">{category}</div>
                    <div class="insight-value" style="color: {color};">{percentage:.1f}%</div>
                    <div style="color: #666; font-size: 12px;">{count:,} mentions</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.divider()
        
        # --- PRIORITY MATRIX ---
        st.markdown("### 🎯 Priority Matrix: Impact vs Frequency")
        
        matrix_df = calculate_priority_matrix(issue_counts, df_filtered)
        
        if not matrix_df.empty and matrix_df['Frequency'].sum() > 0:
            fig_matrix = px.scatter(
                matrix_df,
                x="Frequency",
                y="Impact",
                size="Frequency",
                color="Category",
                color_discrete_map={cat: ISSUE_TAXONOMY[cat]["color"] for cat in ISSUE_TAXONOMY.keys()},
                hover_data=["Category"],
                size_max=60
            )
            
            # Add quadrant labels
            max_freq = matrix_df['Frequency'].max() if matrix_df['Frequency'].max() > 0 else 100
            max_impact = 100
            
            fig_matrix.add_annotation(x=max_freq*0.75, y=75, text="🔴 CRITICAL", showarrow=False, font=dict(size=14, color="#FF6B6B"))
            fig_matrix.add_annotation(x=max_freq*0.25, y=75, text="⚠️ MONITOR", showarrow=False, font=dict(size=14, color="#FFD93D"))
            fig_matrix.add_annotation(x=max_freq*0.75, y=25, text="📋 BACKLOG", showarrow=False, font=dict(size=14, color="#45B7D1"))
            fig_matrix.add_annotation(x=max_freq*0.25, y=25, text="✅ LOW PRIORITY", showarrow=False, font=dict(size=14, color="#4ECDC4"))
            
            fig_matrix.update_layout(
                xaxis_title="Frequency (Number of Mentions)",
                yaxis_title="Impact (% Negative Sentiment)",
                margin=dict(t=30, b=0, l=0, r=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                modebar=dict(bgcolor='rgba(0,0,0,0)', color='white'),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            # Add quadrant lines
            fig_matrix.add_hline(y=50, line_dash="dash", line_color="#4c4c4c")
            fig_matrix.add_vline(x=max_freq/2, line_dash="dash", line_color="#4c4c4c")
            
            st.plotly_chart(fig_matrix, use_container_width=True)
        else:
            st.info("Not enough data to generate priority matrix.")
        
        st.divider()
        
        # --- AI RECOMMENDATIONS ---
        st.markdown("### 💡 AI-Powered Recommendations")
        
        recommendations = generate_recommendations(issue_counts, len(df_filtered))
        
        col1, col2 = st.columns(2)
        
        with col1:
            if recommendations["high"]:
                st.markdown("#### 🔴 High Priority Actions")
                for rec in recommendations["high"]:
                    st.markdown(f"""
                    <div class="recommendation-card priority-high">
                        <strong>{rec['category']}</strong> — {rec['percentage']:.1f}% of reviews ({rec['count']:,} mentions)<br>
                        <span style="color: #b0b3b8;">→ {rec['action']}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("✅ No critical issues detected!")
        
        with col2:
            if recommendations["medium"]:
                st.markdown("#### 🟡 Medium Priority")
                for rec in recommendations["medium"]:
                    st.markdown(f"""
                    <div class="recommendation-card priority-medium">
                        <strong>{rec['category']}</strong> — {rec['percentage']:.1f}% ({rec['count']:,} mentions)<br>
                        <span style="color: #b0b3b8;">→ {rec['action']}</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            if recommendations["opportunities"]:
                st.markdown("#### 🟢 Opportunities")
                for rec in recommendations["opportunities"]:
                    st.markdown(f"""
                    <div class="recommendation-card priority-low">
                        <strong>{rec['category']}</strong> — {rec['percentage']:.1f}% ({rec['count']:,} mentions)<br>
                        <span style="color: #b0b3b8;">→ {rec['action']}</span>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.divider()
        
        # --- COMPETITIVE BENCHMARKING ---
        if selected_game != "All Games" and df is not None:
            st.markdown("### 📊 Competitive Benchmarking")
            st.markdown(f"*How does **{selected_game}** compare to the industry average?*")
            
            benchmark_df = calculate_benchmark(df_filtered, df)
            
            if not benchmark_df.empty:
                # Create benchmark visualization
                fig_bench = px.bar(
                    benchmark_df,
                    x="Category",
                    y=["This Game", "All Games Avg"],
                    barmode="group",
                    color_discrete_sequence=["#66c0f4", "#4c4c4c"],
                    text_auto='.1f'
                )
                fig_bench.update_layout(
                    xaxis_title="",
                    yaxis_title="Issue Mention Rate (%)",
                    margin=dict(t=30, b=0, l=0, r=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white"),
                    modebar=dict(bgcolor='rgba(0,0,0,0)', color='white'),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_bench, use_container_width=True)
                
                # Status summary
                bench_cols = st.columns(4)
                for idx, row in benchmark_df.iterrows():
                    with bench_cols[idx]:
                        diff_text = f"{row['Difference']:+.1f}%" if row['Difference'] != 0 else "0%"
                        st.markdown(f"""
                        <div style="text-align: center; padding: 10px;">
                            <div style="font-size: 24px;">{row['Status']}</div>
                            <div style="color: #b0b3b8; font-size: 12px;">{diff_text} vs avg</div>
                        </div>
                        """, unsafe_allow_html=True)
            
            st.divider()
        
        # --- ISSUE TREND OVER TIME ---
        st.markdown("### 📈 Issue Trend Analysis")
        
        trend_category = st.selectbox("Select category to analyze:", list(ISSUE_TAXONOMY.keys()), key="trend_cat")
        
        trend_data = calculate_issue_trends(df_filtered, trend_category)
        
        if trend_data is not None and len(trend_data) > 1:
            fig_trend = px.line(
                trend_data,
                x="Month",
                y="Issue Rate",
                markers=True,
                color_discrete_sequence=[ISSUE_TAXONOMY[trend_category]["color"]]
            )
            fig_trend.update_layout(
                xaxis_title="",
                yaxis_title="Issue Rate (%)",
                margin=dict(t=30, b=0, l=0, r=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                modebar=dict(bgcolor='rgba(0,0,0,0)', color='white')
            )
            st.plotly_chart(fig_trend, use_container_width=True)
            
            # Trend insight
            if len(trend_data) >= 2:
                first_rate = trend_data.iloc[0]['Issue Rate']
                last_rate = trend_data.iloc[-1]['Issue Rate']
                change = last_rate - first_rate
                if change > 2:
                    st.warning(f"⚠️ **Trending Up:** {trend_category} issues increased by {change:.1f}% over the period")
                elif change < -2:
                    st.success(f"✅ **Improving:** {trend_category} issues decreased by {abs(change):.1f}% over the period")
                else:
                    st.info(f"📊 **Stable:** {trend_category} issues remained relatively constant")
        else:
            st.info("📅 Not enough time-series data available for trend analysis.")
        
        st.divider()
        
        # --- HIGHLIGHTED PLAYER QUOTES ---
        st.markdown("### 💬 Voice of the Player")
        st.markdown("*Most representative feedback from players*")
        
        quote_category = st.selectbox("Select issue category:", list(ISSUE_TAXONOMY.keys()), key="quote_cat")
        
        quotes = get_highlighted_quotes(df_filtered, quote_category, n=5)
        
        if quotes:
            for i, quote in enumerate(quotes, 1):
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #1e1f26 0%, #262730 100%); padding: 16px; border-radius: 12px; margin-bottom: 12px; border-left: 4px solid {quote['color']};">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                        <span style="color: #666; font-size: 12px;">Quote #{i}</span>
                        <span style="font-size: 12px; padding: 2px 8px; border-radius: 4px; background-color: rgba(255,255,255,0.1);">{quote['sentiment']}</span>
                    </div>
                    <div style="color: #e0e0e0; font-style: italic;">"{quote['text']}..."</div>
                    <div style="color: #666; font-size: 11px; margin-top: 8px;">🔑 {quote['relevance']} keyword match(es)</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info(f"No reviews found matching {quote_category} keywords.")
    else:
        st.warning("No data available for analysis.")

# --- TAB 4: LIVE AI LAB ---
with tab4:
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
                with st.spinner("🤖 Analyzing sentiment..."):
                    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
                    with torch.no_grad():
                        logits = model(**inputs).logits
                    probs = F.softmax(logits, dim=-1)
                    score = torch.argmax(probs, dim=-1).item()
                    confidence = probs[0][score].item()
                
                labels = {0: ("Dissatisfied", "🔴"), 1: ("Neutral", "⚪"), 2: ("Satisfied", "🟢")}
                label_txt, icon = labels[score]
                
                st.markdown(f"""
                <div class="result-card fade-in">
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

# --- TAB 5: BENCHMARKS ---
with tab5:
    st.subheader("🏆 Model Performance Leaderboard")
    df_bench = load_benchmark()
    if df_bench is not None:
        fig = px.bar(
            df_bench.sort_values("Accuracy", ascending=True), 
            x="Accuracy", y="Model", orientation='h', text_auto='.2%',
            color="Accuracy", color_continuous_scale="Viridis"
        )
        fig.update_layout(
            margin=dict(t=30, b=0, l=0, r=0),
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)", 
            font=dict(color="white"),
            modebar=dict(bgcolor='rgba(0,0,0,0)', color='white')
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_bench)

# --- TAB 6: GAME COMPARISON LAB ---
with tab6:
    st.subheader("⚔️ Game Comparison Lab")
    st.markdown("*Compare sentiment and issues across multiple games*")
    
    if df is not None:
        game_list_compare = df['app_name'].unique().tolist()
        
        col1, col2 = st.columns(2)
        with col1:
            game_a = st.selectbox("Select Game A:", game_list_compare, key="compare_a")
        with col2:
            game_b = st.selectbox("Select Game B:", [g for g in game_list_compare if g != game_a], key="compare_b")
        
        if game_a and game_b:
            df_a = df[df['app_name'] == game_a]
            df_b = df[df['app_name'] == game_b]
            
            st.divider()
            
            # Sentiment comparison
            st.markdown("### 📊 Sentiment Distribution Comparison")
            
            comp_cols = st.columns(2)
            
            with comp_cols[0]:
                st.markdown(f"**{game_a}**")
                sat_a = (df_a['sentiment'] == 2).mean() * 100
                neu_a = (df_a['sentiment'] == 1).mean() * 100
                dis_a = (df_a['sentiment'] == 0).mean() * 100
                
                fig_a = px.pie(
                    values=[sat_a, neu_a, dis_a],
                    names=['Satisfied', 'Neutral', 'Dissatisfied'],
                    color_discrete_sequence=['#5DBF82', '#F5E85C', '#E07B53'],
                    hole=0.5
                )
                fig_a.update_layout(
                    showlegend=False,
                    margin=dict(t=0, b=0, l=0, r=0),
                    paper_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_a, use_container_width=True)
                st.metric("Satisfaction Rate", f"{sat_a:.1f}%", f"{len(df_a):,} reviews")
            
            with comp_cols[1]:
                st.markdown(f"**{game_b}**")
                sat_b = (df_b['sentiment'] == 2).mean() * 100
                neu_b = (df_b['sentiment'] == 1).mean() * 100
                dis_b = (df_b['sentiment'] == 0).mean() * 100
                
                fig_b = px.pie(
                    values=[sat_b, neu_b, dis_b],
                    names=['Satisfied', 'Neutral', 'Dissatisfied'],
                    color_discrete_sequence=['#5DBF82', '#F5E85C', '#E07B53'],
                    hole=0.5
                )
                fig_b.update_layout(
                    showlegend=False,
                    margin=dict(t=0, b=0, l=0, r=0),
                    paper_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_b, use_container_width=True)
                st.metric("Satisfaction Rate", f"{sat_b:.1f}%", f"{len(df_b):,} reviews")
            
            st.divider()
            
            # Issue comparison
            st.markdown("### 🔍 Issue Category Comparison")
            
            issues_a, _ = extract_issues(df_a)
            issues_b, _ = extract_issues(df_b)
            
            compare_data = []
            for cat in ISSUE_TAXONOMY.keys():
                pct_a = (issues_a[cat] / len(df_a) * 100) if len(df_a) > 0 else 0
                pct_b = (issues_b[cat] / len(df_b) * 100) if len(df_b) > 0 else 0
                compare_data.append({"Category": cat, "Game": game_a, "Issue Rate": pct_a})
                compare_data.append({"Category": cat, "Game": game_b, "Issue Rate": pct_b})
            
            compare_df = pd.DataFrame(compare_data)
            
            fig_compare = px.bar(
                compare_df,
                x="Category",
                y="Issue Rate",
                color="Game",
                barmode="group",
                color_discrete_sequence=["#E07B53", "#6B9DD8"]
            )
            fig_compare.update_layout(
                margin=dict(t=30, b=0, l=0, r=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                yaxis_title="Issue Mention Rate (%)"
            )
            st.plotly_chart(fig_compare, use_container_width=True)
            
            # Winner summary
            st.divider()
            st.markdown("### 🏆 Comparison Summary")
            
            winner_sat = game_a if sat_a > sat_b else game_b
            diff_sat = abs(sat_a - sat_b)
            
            st.markdown(f"""
            <div class="insight-card" style="border-left: 4px solid #5DBF82;">
                <strong>Satisfaction Winner:</strong> <span style="color: #5DBF82;">{winner_sat}</span><br>
                <span style="color: var(--muted-foreground);">Leading by {diff_sat:.1f} percentage points</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("No data available for comparison.")

# --- TAB 7: SENTIMENT PREDICTOR ---
with tab7:
    st.subheader("🔮 Sentiment Predictor")
    st.markdown("*Predict likely sentiment based on game features and keywords*")
    
    if df is not None:
        st.markdown("### Enter game characteristics to predict sentiment:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            has_multiplayer = st.checkbox("Has Multiplayer", value=False)
            has_mtx = st.checkbox("Has Microtransactions", value=False)
            is_early_access = st.checkbox("Early Access", value=False)
            has_dlc = st.checkbox("Has DLC", value=False)
        
        with col2:
            genre = st.selectbox("Primary Genre:", ["Action", "RPG", "Strategy", "Simulation", "Indie", "Adventure", "Sports"])
            price_range = st.selectbox("Price Range:", ["Free-to-Play", "Under $10", "$10-$30", "$30-$60", "Over $60"])
        
        keywords_input = st.text_input("Key features (comma-separated):", placeholder="e.g., open world, story-driven, roguelike")
        
        if st.button("🔮 Predict Sentiment", type="primary", use_container_width=True):
            # Simple heuristic-based prediction model
            base_satisfaction = 65.0  # Average baseline
            
            # Adjust based on features
            if has_multiplayer:
                base_satisfaction += 2  # Slight boost
            if has_mtx:
                base_satisfaction -= 8  # Negative impact
            if is_early_access:
                base_satisfaction -= 5  # Early access penalty
            if has_dlc:
                base_satisfaction -= 2  # Slight concern about DLC
            
            # Genre adjustments (based on typical Steam patterns)
            genre_adj = {"Indie": 5, "RPG": 3, "Strategy": 2, "Action": 0, "Simulation": 1, "Adventure": 2, "Sports": -3}
            base_satisfaction += genre_adj.get(genre, 0)
            
            # Price adjustments
            price_adj = {"Free-to-Play": -5, "Under $10": 5, "$10-$30": 3, "$30-$60": 0, "Over $60": -5}
            base_satisfaction += price_adj.get(price_range, 0)
            
            # Keyword analysis
            if keywords_input:
                positive_keywords = ["story", "beautiful", "fun", "polish", "quality", "innovative"]
                negative_keywords = ["grind", "bug", "broken", "expensive", "p2w", "repetitive"]
                
                keywords = [k.strip().lower() for k in keywords_input.split(",")]
                for kw in keywords:
                    if any(pk in kw for pk in positive_keywords):
                        base_satisfaction += 3
                    if any(nk in kw for nk in negative_keywords):
                        base_satisfaction -= 5
            
            # Clamp to valid range
            predicted_sat = max(min(base_satisfaction, 95), 20)
            predicted_neu = min(25, 100 - predicted_sat)
            predicted_dis = 100 - predicted_sat - predicted_neu
            
            st.divider()
            st.markdown("### 📊 Predicted Sentiment Breakdown")
            
            pred_cols = st.columns(3)
            with pred_cols[0]:
                st.markdown(f"""
                <div class="insight-card" style="border-top: 3px solid #5DBF82; text-align: center;">
                    <div style="font-size: 36px;">🟢</div>
                    <div class="insight-value" style="color: #5DBF82;">{predicted_sat:.0f}%</div>
                    <div class="insight-header">SATISFIED</div>
                </div>
                """, unsafe_allow_html=True)
            with pred_cols[1]:
                st.markdown(f"""
                <div class="insight-card" style="border-top: 3px solid #F5E85C; text-align: center;">
                    <div style="font-size: 36px;">⚪</div>
                    <div class="insight-value" style="color: #F5E85C;">{predicted_neu:.0f}%</div>
                    <div class="insight-header">NEUTRAL</div>
                </div>
                """, unsafe_allow_html=True)
            with pred_cols[2]:
                st.markdown(f"""
                <div class="insight-card" style="border-top: 3px solid #E07B53; text-align: center;">
                    <div style="font-size: 36px;">🔴</div>
                    <div class="insight-value" style="color: #E07B53;">{predicted_dis:.0f}%</div>
                    <div class="insight-header">DISSATISFIED</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Risk factors
            st.divider()
            st.markdown("### ⚠️ Risk Factors Detected")
            risks = []
            if has_mtx:
                risks.append("💰 Microtransactions often receive negative sentiment")
            if is_early_access:
                risks.append("🚧 Early Access games face higher scrutiny for bugs")
            if price_range == "Over $60":
                risks.append("💵 Premium pricing increases expectations")
            if has_mtx and price_range not in ["Free-to-Play", "Under $10"]:
                risks.append("🔴 Paid game with MTX is a high-risk combination")
            
            if risks:
                for risk in risks:
                    st.warning(risk)
            else:
                st.success("✅ No major risk factors detected!")
    else:
        st.warning("No data available.")

# --- TAB 8: REVIEW EXPLORER ---
with tab8:
    st.subheader("📰 Review Explorer")
    st.markdown("*Search and filter player reviews*")
    
    if df is not None:
        # Filters
        filter_cols = st.columns([2, 1, 1, 1])
        
        with filter_cols[0]:
            search_query = st.text_input("🔍 Search reviews:", placeholder="Enter keywords...")
        
        with filter_cols[1]:
            sentiment_filter = st.selectbox("Sentiment:", ["All", "Satisfied", "Neutral", "Dissatisfied"])
        
        with filter_cols[2]:
            game_filter = st.selectbox("Game:", ["All Games"] + df['app_name'].unique().tolist(), key="explorer_game")
        
        with filter_cols[3]:
            sort_by = st.selectbox("Sort by:", ["Newest", "Oldest", "Longest", "Shortest"])
        
        # Apply filters
        filtered_df = df.copy()
        
        if search_query:
            filtered_df = filtered_df[filtered_df['clean_text'].str.contains(search_query, case=False, na=False)]
        
        if sentiment_filter != "All":
            sent_map = {"Satisfied": 2, "Neutral": 1, "Dissatisfied": 0}
            filtered_df = filtered_df[filtered_df['sentiment'] == sent_map[sentiment_filter]]
        
        if game_filter != "All Games":
            filtered_df = filtered_df[filtered_df['app_name'] == game_filter]
        
        # Add length column for sorting
        filtered_df['text_length'] = filtered_df['clean_text'].astype(str).apply(len)
        
        # Sort
        if sort_by == "Newest" and 'date' in filtered_df.columns:
            filtered_df = filtered_df.sort_values('date', ascending=False)
        elif sort_by == "Oldest" and 'date' in filtered_df.columns:
            filtered_df = filtered_df.sort_values('date', ascending=True)
        elif sort_by == "Longest":
            filtered_df = filtered_df.sort_values('text_length', ascending=False)
        elif sort_by == "Shortest":
            filtered_df = filtered_df.sort_values('text_length', ascending=True)
        
        st.divider()
        st.markdown(f"**Found {len(filtered_df):,} reviews**")
        
        # Pagination
        reviews_per_page = 10
        total_pages = max(1, (len(filtered_df) - 1) // reviews_per_page + 1)
        page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
        
        start_idx = (page - 1) * reviews_per_page
        end_idx = start_idx + reviews_per_page
        
        # Display reviews
        for idx, row in filtered_df.iloc[start_idx:end_idx].iterrows():
            sent = row.get('sentiment', 1)
            sent_emoji = {0: "🔴", 1: "⚪", 2: "🟢"}.get(sent, "⚪")
            sent_color = {0: "#E07B53", 1: "#F5E85C", 2: "#5DBF82"}.get(sent, "#F5E85C")
            text = str(row.get('clean_text', ''))[:500]
            game = row.get('app_name', 'Unknown')
            
            st.markdown(f"""
            <div class="insight-card" style="border-left: 4px solid {sent_color};">
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                    <span style="font-size: 12px; color: var(--muted-foreground);">🎮 {game}</span>
                    <span>{sent_emoji}</span>
                </div>
                <div style="color: var(--foreground);">{text}{'...' if len(str(row.get('clean_text', ''))) > 500 else ''}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Export option
        st.divider()
        if st.button("📥 Export Filtered Reviews to CSV"):
            csv = filtered_df[['app_name', 'clean_text', 'sentiment']].to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="filtered_reviews.csv",
                mime="text/csv"
            )
    else:
        st.warning("No data available.")