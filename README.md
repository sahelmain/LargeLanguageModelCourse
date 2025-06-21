# Text Classification: Human vs AI-Written Text 🤖📝

**Project**: Text Mining & Machine Learning  
**Task**: Binary text classification to distinguish between human-written and AI-generated text  
**Accuracy Achieved**: **96.25%** with SVM (Sigmoid kernel)

## 📋 Project Overview

This project implements a complete machine learning pipeline to classify text as either human-written (label 0) or AI-generated (label 1). The solution includes comprehensive text preprocessing, feature extraction, model development, and evaluation.

### 🎯 Key Results
- **Final Accuracy**: 96.25% on test set
- **Cross-Validation**: 96.28% ± 0.8%
- **F1-Score**: 0.96
- **AUC**: 0.9949 (Excellent performance)

## 📁 Repository Structure

```
├── assignment1_text_classification.ipynb  # Main notebook with complete implementation
├── Final_predictions.csv                  # Predictions for test data (869 samples)
├── AI_vs_human_train_dataset.csv         # Training dataset (3,728 samples)
├── Final_test_data.csv                   # Test dataset (869 samples, no labels)
├── project.txt                       # project requirements
└── README.md                            # Project documentation
```

## 🔧 Technical Implementation

### Task 1: Data Exploration (10 points)
- **Dataset**: 3,728 perfectly balanced essays (50% human, 50% AI)
- **Analysis**: Statistical overview and class distribution visualization
- **Sample texts**: Representative examples from each category

### Task 2: Text Preprocessing (30 points)
- **Stop Words Removal**: 22.7% vocabulary reduction (26,228 → 20,280 words)
- **Stemming**: Porter, Snowball, and Lancaster stemmer comparison
- **Lemmatization**: WordNet lemmatizer with POS tagging for semantic accuracy

### Task 3: Feature Extraction (20 points)
- **TF-IDF Vectorization**: Term frequency with inverse document frequency weighting
- **Bag of Words**: Count-based vectorization
- **N-grams**: Unigrams, bigrams, and trigrams analysis
- **Best Method**: TF-IDF with bigrams (96.60% accuracy)

### Task 4: Machine Learning Models (30 points)

#### SVM Implementation
- **Kernels Tested**: Linear, RBF, Polynomial, Sigmoid
- **Best Performance**: Sigmoid kernel (96.25% accuracy)
- **Parameters**: C=1.0, optimized for text classification

#### Decision Tree Implementation  
- **Criteria**: Gini vs Entropy comparison
- **Depth Analysis**: Various max_depth settings
- **Best Performance**: Gini with max_depth=10 (86.70% CV score)

### Task 5: Model Evaluation (10 points)
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-score
- **Confusion Matrix**: Detailed misclassification analysis
- **ROC Curve**: AUC = 0.9949 indicating excellent discrimination
- **Error Analysis**: Pattern identification in misclassified examples

## 🏆 Model Comparison

| Model | Accuracy | CV Score | F1-Score | Training Time |
|-------|----------|----------|----------|---------------|
| **SVM (Sigmoid)** | **96.25%** | **96.28% ± 0.8%** | **0.96** | 1.36s |
| SVM (Linear) | 96.34% | 96.21% ± 0.6% | 0.96 | 1.43s |
| Decision Tree | 82.93% | 86.70% ± 1.7% | 0.83 | 0.16s |

## 📊 Key Findings

### Feature Engineering Impact
- **TF-IDF vs BoW**: TF-IDF consistently outperformed Bag of Words
- **N-grams**: Bigrams captured valuable context, improving performance
- **Parameter Tuning**: `min_df=3`, `max_df=0.92`, `max_features=4500` proved optimal

### Text Patterns Discovered
- **AI Text**: Tends to use more varied vocabulary and sophisticated structures
- **Human Text**: Shows more natural variability and occasional irregularities
- **Misclassifications**: Only 3.6% error rate with clear pattern distinctions

## 🛠️ Technologies Used

- **Python 3.x**
- **Pandas & NumPy**: Data manipulation and analysis
- **NLTK**: Natural language processing and text preprocessing
- **Scikit-learn**: Machine learning models and evaluation
- **Matplotlib & Seaborn**: Data visualization
- **Jupyter Notebook**: Interactive development environment

## 🚀 How to Run

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd text-classification-assignment
   ```

2. **Install dependencies**:
   ```bash
   pip install pandas numpy scikit-learn nltk matplotlib seaborn jupyter
   ```

3. **Download NLTK data**:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   nltk.download('averaged_perceptron_tagger')
   ```

4. **Run the notebook**:
   ```bash
   jupyter notebook assignment1_text_classification.ipynb
   ```

## 📈 Results

### Final Performance Metrics
- **Test Accuracy**: 96.25%
- **Precision**: 0.96 (both classes)
- **Recall**: 0.96 (both classes)
- **F1-Score**: 0.96
- **Cross-Validation**: 96.28% ± 0.82%

### Confusion Matrix (Best Model)
```
Predicted:    Human  AI
Human:         533   27
AI:             15  544
```

### Error Analysis
- **Human misclassified as AI**: 27 cases (often shorter, simpler texts)
- **AI misclassified as Human**: 15 cases (more sophisticated AI writing)
- **Total Error Rate**: 3.75% (42/1119)

## 📝 Project Requirements Fulfilled

✅ **Data Exploration**: Complete statistical analysis and visualization  
✅ **Text Preprocessing**: Stop words, stemming, lemmatization with examples  
✅ **Feature Extraction**: TF-IDF and BoW with n-gram analysis  
✅ **SVM Implementation**: Multiple kernels tested and optimized  
✅ **Decision Tree**: Various parameters and criteria compared  
✅ **Model Evaluation**: Comprehensive metrics and error analysis  
✅ **Final Predictions**: 869 test predictions generated and saved  


## 📄 License

This project is for educational purposes as part of a text mining course assignment. 
