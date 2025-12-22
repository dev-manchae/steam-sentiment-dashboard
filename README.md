# 🎮 Steam Sentiment Intelligence

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)

**Deep learning-powered sentiment analysis dashboard for Steam game reviews using RoBERTa transformers.**

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [Usage](#-usage) • [Model](#-model) • [Architecture](#-architecture)

</div>

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📊 **Analytics Dashboard** | Interactive visualizations including sentiment distribution, trends over time, and game leaderboards |
| ☁️ **Topic Word Clouds** | Dynamic word clouds for satisfied vs. dissatisfied reviews with customizable settings |
| 🔍 **Developer Insights** | AI-powered issue categorization, priority matrix, benchmarking, trend analysis, and actionable recommendations |
| 🤖 **Live AI Lab** | Real-time sentiment prediction using your own custom text input with loading animations |
| 🏆 **Model Benchmarks** | Performance comparison against other NLP models (SVM, VADER, Naive Bayes) |
| ⚔️ **Game Comparison** | Side-by-side comparison of sentiment and issues across multiple games |
| 🔮 **Sentiment Predictor** | Predict likely sentiment based on game features, genre, and pricing |
| 📰 **Review Explorer** | Search, filter, and export player reviews with pagination |
| 🎨 **Steam-themed UI** | Beautiful dark theme with smooth loading animations and transitions |

---

## 🖼️ Demo

The dashboard includes **8 main tabs**:

1. **📊 Dashboard** - Pie charts, area charts, box plots, and leaderboards
2. **☁️ Topic Clouds** - Word frequency visualizations by sentiment
3. **🔍 Dev Insights** - Developer intelligence with issue categorization and recommendations
4. **🤖 AI Lab** - Test the model with custom game reviews
5. **🏆 Benchmarks** - Compare RoBERTa against baseline models
6. **⚔️ Compare Games** - Side-by-side game sentiment comparison
7. **🔮 Predictor** - Predict sentiment based on game features
8. **📰 Explorer** - Search and filter reviews database

---

## 🚀 Installation

### Prerequisites

- Python 3.11+
- pip package manager

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/steam-sentiment-dashboard.git
   cd steam-sentiment-dashboard
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open in browser**
   Navigate to `http://localhost:8501`

### Using Dev Containers (Codespaces)

This project includes a `.devcontainer` configuration for GitHub Codespaces:

- Python 3.11 environment pre-configured
- Auto-installs requirements
- Auto-launches Streamlit on port 8501

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web application framework |
| `pandas` | Data manipulation and analysis |
| `plotly` | Interactive visualizations |
| `wordcloud` | Word cloud generation |
| `matplotlib` | Static plotting |
| `transformers` | Hugging Face transformers library |
| `torch` | PyTorch deep learning framework |

---

## 🧠 Model

### RoBERTa Sentiment Classifier

The dashboard uses a fine-tuned **RoBERTa-base** model hosted on Hugging Face:

- **Model**: [`manchae86/steam-review-roberta`](https://huggingface.co/manchae86/steam-review-roberta)
- **Base Architecture**: RoBERTa (Robustly Optimized BERT Pretraining Approach)
- **Task**: 3-class sentiment classification
- **Labels**: `Dissatisfied (0)`, `Neutral (1)`, `Satisfied (2)`

### Benchmark Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| **RoBERTa-Teacher** | **91.08%** | 90.99% | 91.08% | 90.99% |
| DistilBERT-Teacher | 89.92% | 89.82% | 89.92% | 89.85% |
| SVM | 80.24% | 81.61% | 80.24% | 80.62% |
| VADER | 75.23% | 75.80% | 75.23% | 75.00% |
| Naive Bayes | 74.36% | 74.27% | 74.36% | 71.94% |

---

## 🏗️ Architecture

```
steam-sentiment-dashboard/
├── .devcontainer/
│   └── devcontainer.json       # GitHub Codespaces configuration
├── data/
│   ├── benchmark.csv           # Model comparison metrics
│   └── steam_reviews.csv       # Training/analysis dataset (gitignored)
├── model/                      # Local model files (gitignored)
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── ...
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── .gitignore
└── README.md
```

### Data Sources

- **Reviews Dataset**: Hosted on Hugging Face - automatically fetched from [`manchae86/steam-review-roberta`](https://huggingface.co/manchae86/steam-review-roberta/resolve/main/steam_reviews.csv)
- **Benchmark Data**: Local CSV file with model comparison metrics

---

## ⚙️ Configuration

The application uses sensible defaults but can be customized:

| Setting | Default | Description |
|---------|---------|-------------|
| Port | `8501` | Streamlit server port |
| Max Token Length | `256` | Maximum input sequence length for inference |
| Layout | `wide` | Full-width dashboard layout |

---

## 🎯 Usage Guide

### Sidebar Controls

1. **Game Selection** - Filter analytics by specific game or view all games
2. **Metrics Display** - Shows total reviews and satisfaction rate for selected context

### Tab Navigation

#### Tab 1: 📊 Analytics Dashboard
- View sentiment distribution pie chart
- Explore temporal trends with area charts
- Compare games on the leaderboard (when viewing "All Games")
- Analyze review length patterns

#### Tab 2: ☁️ Topic Clouds
- Toggle between Satisfied/Dissatisfied perspectives
- Automatically filters common gaming terms
- Uses Viridis (positive) and Magma (negative) color schemes

#### Tab 3: 🔍 Developer Insights
- **Executive Summary** - 4 issue category cards (Technical, Gameplay, Content, Monetization)
- **Priority Matrix** - Interactive scatter plot with Impact vs Frequency
- **AI Recommendations** - High/Medium priority actions and opportunities
- **Competitive Benchmarking** - Compare selected game vs all-games average
- **Issue Trend Analysis** - Track how issues change over time
- **Voice of the Player** - Representative quotes with sentiment badges

#### Tab 4: 🤖 Live AI Lab
- Enter custom review text
- Click "Analyze Sentiment" to get predictions
- View confidence scores with progress bar
- Loading animations during inference

#### Tab 5: 🏆 Model Benchmarks
- Compare RoBERTa against baseline models
- Interactive bar chart with accuracy metrics
- Full metrics table available

#### Tab 6: ⚔️ Game Comparison Lab
- Select two games for side-by-side comparison
- Compare sentiment distribution pie charts
- Issue category comparison bar chart
- Winner summary with difference calculation

#### Tab 7: 🔮 Sentiment Predictor
- Input game features (multiplayer, MTX, early access, DLC)
- Select genre and price range
- Get predicted sentiment breakdown (Satisfied/Neutral/Dissatisfied)
- View risk factors and warnings

#### Tab 8: 📰 Review Explorer
- Search reviews by keyword
- Filter by sentiment, game, and sort order
- Paginated review display (10 per page)
- Export filtered results to CSV

---

## 🎨 UI Features

- **Steam-inspired dark theme** with blue gradient background
- **Smooth loading animations** with shimmer skeleton placeholders
- **Fade-in transitions** on cards with staggered delays
- **Loading spinners** during data processing
- **Responsive layout** for different screen sizes

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

This project is open source and available under the MIT License.

---