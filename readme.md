# Real-Time Sentiment Analysis for Brand Perception

A comprehensive Natural Language Processing (NLP) project for analyzing social media sentiment patterns using Twitter data. This project leverages machine learning and sentiment analysis techniques to extract insights from social media content and understand public perception toward brands and entities.

## 👨‍💻 Author

**Ayush Amin** (22BCE2081)  
📧 ayushnamin@gmail.com

## 📋 Table of Contents

- [Introduction](#introduction)
- [Business Problem](#business-problem)
- [Objectives](#objectives)
- [Features](#features)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Recommendations](#recommendations)
- [Future Enhancements](#future-enhancements)

## 🎯 Introduction

In today's digital age, understanding public sentiment and attitudes towards specific topics or brands on social media is crucial for businesses and organizations. Social media platforms serve as vast repositories of unstructured text data, and unlocking the insights hidden within this data can provide a significant competitive advantage.

This project utilizes Natural Language Processing (NLP) and data analytics to gain comprehensive insights into public sentiment and attitudes on Twitter, enabling data-driven decision-making for businesses.

## 💼 Business Problem

The primary business challenge addressed in this project is to analyze and understand sentiment patterns on social media platforms, specifically Twitter. By performing sentiment analysis at scale, businesses can:

- **Monitor brand reputation** in real-time
- **Identify customer pain points** and areas of satisfaction
- **Make informed decisions** based on public perception
- **Engage with their audience effectively**
- **Adapt marketing strategies** based on sentiment trends
- **Benchmark against competitors**

## 🎯 Objectives

1. **Analyze sentiment patterns**: Utilize advanced NLP techniques to perform sentiment analysis on Twitter text data
2. **Visualize sentiment trends**: Create informative data visualizations to represent sentiment distribution and trends
3. **Extract actionable insights**: Dive deep into sentiment analysis results to extract valuable business insights
4. **Build predictive models**: Develop machine learning models for automated sentiment classification
5. **Provide recommendations**: Offer data-driven recommendations for brand management and engagement strategies

## ✨ Features

- 📊 **Comprehensive EDA**: Detailed exploratory data analysis with visualizations
- 🧹 **Advanced Text Preprocessing**: Robust NLP preprocessing pipeline including tokenization, lemmatization, and stopwords removal
- 💭 **Multi-Method Sentiment Analysis**: 
  - VADER sentiment analysis for rule-based classification
  - Machine learning-based classification using Logistic Regression
- 📈 **Rich Visualizations**: Sentiment distribution, entity analysis, word clouds, and text length analysis
- 🤖 **Machine Learning Model**: TF-IDF vectorization with Logistic Regression classifier
- 🔍 **Entity-Based Analysis**: Sentiment breakdown by specific brands and entities
- 📉 **Performance Metrics**: Accuracy scores and classification reports

## 📊 Dataset

The project uses Twitter Entity Sentiment Analysis dataset from Kaggle:

**Source**: [Twitter Entity Sentiment Analysis - Kaggle](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis)

### Dataset Statistics:
- **Training Set**: 74,682 tweets
- **Validation Set**: 1,758 tweets
- **Total Records**: 76,440 tweets

### Data Schema:
| Column | Description | Type |
|--------|-------------|------|
| ID | Unique identifier for each tweet | Integer |
| Entity | Brand or entity mentioned in the tweet | String |
| Sentiment | Sentiment label (Positive/Negative/Neutral) | Categorical |
| Content | The actual tweet text | String |

### Sentiment Classes:
- **Positive**: Favorable sentiment towards the entity
- **Negative**: Unfavorable sentiment towards the entity
- **Neutral**: Neutral or irrelevant sentiment

## 🛠️ Tech Stack

### Programming Language
- **Python 3.x**

### Core Libraries

#### Data Manipulation & Analysis
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing

#### Visualization
- `matplotlib` - Plotting and visualization
- `seaborn` - Statistical data visualization
- `wordcloud` - Word cloud generation

#### Natural Language Processing
- `nltk` - Natural Language Toolkit
  - `SentimentIntensityAnalyzer` (VADER)
  - `punkt` tokenizer
  - `stopwords` corpus
  - `WordNetLemmatizer`
  - `wordnet` corpus
- `re` - Regular expressions for text cleaning

#### Machine Learning
- `scikit-learn`
  - `TfidfVectorizer` - Feature extraction
  - `LogisticRegression` - Classification model
  - `train_test_split` - Data splitting
  - `classification_report` - Model evaluation
  - `accuracy_score` - Performance metrics

## 📁 Project Structure

```
DS_RealTimeSentimentAnalysis/
├── app.ipynb                    # Main Jupyter notebook with full analysis
├── twitter_training.csv         # Training dataset (74,682 tweets)
├── twitter_validation.csv       # Validation dataset (1,758 tweets)
├── README.md                    # Project documentation
└── readme_old.md                # Legacy documentation
```

## 🚀 Installation

### Prerequisites
- Python 3.7 or higher
- Jupyter Notebook or JupyterLab
- pip package manager

### Setup Instructions

1. **Clone the repository**:
```bash
git clone <repository-url>
cd DS_RealTimeSentimentAnalysis
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install required packages**:
```bash
pip install pandas numpy matplotlib seaborn nltk wordcloud scikit-learn
```

4. **Download NLTK data**:
```python
import nltk
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

5. **Launch Jupyter Notebook**:
```bash
jupyter notebook app.ipynb
```

## 💻 Usage

### Running the Analysis

1. Open `app.ipynb` in Jupyter Notebook
2. Run all cells sequentially to:
   - Load and explore the dataset
   - Perform data cleaning and preprocessing
   - Conduct exploratory data analysis
   - Apply NLP preprocessing
   - Perform sentiment analysis using VADER
   - Train and evaluate the machine learning model
   - Generate visualizations and insights

### Key Code Sections

#### Data Loading
```python
import pandas as pd

col_names = ['ID', 'Entity', 'Sentiment', 'Content']
df = pd.read_csv('twitter_training.csv', names=col_names)
```

#### Text Preprocessing
The notebook includes comprehensive text preprocessing:
- Removing special characters and URLs
- Converting to lowercase
- Tokenization
- Lemmatization
- Stopwords removal

#### Sentiment Analysis with VADER
```python
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()
df['vader_scores'] = df['Content'].apply(lambda x: sia.polarity_scores(x))
```

#### Machine Learning Model
```python
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Feature extraction
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['processed_text'])
y = df['Sentiment']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
```

## 🔬 Methodology

### 1. Data Understanding & Exploration
- Load Twitter dataset with 74,682 training samples
- Examine data structure, types, and distributions
- Identify data quality issues

### 2. Data Preparation
- **Handle missing values**: Drop 686 rows with missing content
- **Remove duplicates**: Eliminate 2,340 duplicate tweets
- **Normalize labels**: Replace "Irrelevant" with "Neutral"
- **Reset indices**: Clean data indexing

### 3. Exploratory Data Analysis (EDA)
- **Sentiment distribution**: Analyze balance of positive, negative, and neutral tweets
- **Entity analysis**: Examine which brands/entities are most discussed
- **Text length analysis**: Study tweet length patterns
- **Word frequency**: Generate word clouds and frequency distributions
- **Entity-sentiment correlation**: Analyze sentiment patterns by entity

### 4. NLP Preprocessing Pipeline
```
Raw Text → Remove Special Characters → Lowercase → Tokenization
    ↓
Stopwords Removal ← Lemmatization ← POS Tagging
    ↓
Cleaned Text
```

**Steps**:
1. Remove URLs, mentions, hashtags, and special characters
2. Convert to lowercase
3. Tokenize text into words
4. Remove stopwords (common words like 'the', 'is', 'at')
5. Lemmatize words to their base form
6. Reconstruct cleaned text

### 5. Sentiment Analysis

#### Rule-Based Approach (VADER)
- Apply VADER SentimentIntensityAnalyzer
- Generate compound sentiment scores
- Categorize into positive/negative/neutral
- Visualize sentiment distributions

#### Machine Learning Approach
- **Feature Engineering**: TF-IDF vectorization (max 5,000 features)
- **Model Selection**: Logistic Regression
- **Training**: Fit model on training data
- **Evaluation**: Test on validation set

### 6. Model Evaluation
- Generate classification reports
- Calculate accuracy scores
- Analyze confusion matrices
- Compare rule-based vs ML-based approaches

### 7. Insights Extraction
- Identify sentiment trends by entity
- Detect potential issues or opportunities
- Generate actionable recommendations

## 📈 Results

The project delivers:

### Visualizations
- 📊 Sentiment distribution bar charts and pie charts
- 🏢 Entity frequency distributions
- 📈 Sentiment trends by entity
- ☁️ Word clouds for each sentiment category
- 📏 Text length distributions
- 🔥 Heatmaps for entity-sentiment correlations

### Model Performance
- **Accuracy Score**: Evaluated on test set
- **Classification Report**: Precision, recall, and F1-scores for each sentiment class
- **Confusion Matrix**: Detailed breakdown of prediction performance

### Key Insights
- Identification of most positively/negatively perceived brands
- Understanding of sentiment patterns across different entities
- Detection of common themes in positive vs negative tweets
- Text characteristics that indicate sentiment

## 💡 Recommendations

Based on the sentiment analysis results, the following recommendations are provided:

### 1. Continuous Monitoring
- Implement **real-time sentiment tracking** for brands and entities
- Set up **alert systems** for sudden negative sentiment spikes
- Monitor sentiment trends over time to identify patterns

### 2. Tailored Engagement Strategies
- **For Positive Sentiment**: Amplify positive mentions, engage with satisfied customers, encourage user-generated content
- **For Negative Sentiment**: Respond promptly to complaints, address issues publicly, implement damage control strategies
- **For Neutral Sentiment**: Create engaging content to shift neutral mentions toward positive

### 3. Content Strategy
- Use sentiment insights to **guide content creation**
- Develop campaigns that resonate with audience sentiment
- Avoid topics or messaging that generate negative sentiment

### 4. Customer Feedback Analysis
- Integrate sentiment analysis into **customer feedback loops**
- Identify recurring complaints or feature requests
- Prioritize product improvements based on sentiment data

### 5. Competitive Benchmarking
- Compare sentiment across competing brands
- Identify **competitive advantages** and weaknesses
- Learn from competitors' successes and failures

### 6. Real-Time Response System
- Implement automated sentiment monitoring
- Enable rapid response to customer concerns
- Build customer loyalty through proactive engagement

### 7. Sentiment-Driven Decision Making
- Use sentiment data for **strategic planning**
- Inform marketing campaign timing and messaging
- Guide product development priorities

## 🚀 Future Enhancements

### Technical Improvements
- [ ] **Deep Learning Models**: Implement LSTM, BERT, or transformer-based models for improved accuracy
- [ ] **Multi-language Support**: Extend analysis to non-English tweets
- [ ] **Real-time Processing**: Build streaming pipeline for live sentiment analysis
- [ ] **Aspect-Based Sentiment**: Analyze sentiment toward specific product features
- [ ] **Emotion Detection**: Classify emotions beyond positive/negative/neutral (anger, joy, fear, etc.)

### Features
- [ ] **Dashboard Development**: Create interactive web dashboard for visualization
- [ ] **API Development**: Build REST API for sentiment prediction
- [ ] **Automated Reporting**: Generate scheduled sentiment reports
- [ ] **Influencer Analysis**: Identify and analyze key opinion leaders
- [ ] **Trend Detection**: Implement anomaly detection for viral content

### Data Expansion
- [ ] **Multi-platform Analysis**: Include data from Facebook, Instagram, Reddit
- [ ] **Image Analysis**: Add visual sentiment analysis from social media images
- [ ] **Historical Analysis**: Incorporate temporal analysis for long-term trends
- [ ] **Geographic Analysis**: Add location-based sentiment mapping

### Model Improvements
- [ ] **Ensemble Methods**: Combine multiple models for better performance
- [ ] **Active Learning**: Implement continuous model improvement
- [ ] **Sarcasm Detection**: Handle sarcastic and ironic statements
- [ ] **Context Awareness**: Improve handling of context-dependent sentiment

## 📝 Notes

- This project was developed as part of a data science course (22BCE2081)
- The dataset is sourced from Kaggle and is used for educational purposes
- VADER sentiment analysis is particularly effective for social media text
- The model performance may vary based on the entity and domain

## 📄 License

This project is for educational and research purposes.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## 📞 Contact

**Ayush Amin**  
Email: ayushnamin@gmail.com  
Student ID: 22BCE2081

---

**Note**: Make sure to download NLTK data packages before running the notebook. The notebook includes automatic download commands, but manual download may be required in some environments.
