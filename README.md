K-Nearest Neighbor Classification - Pattern
Recognition Mini Project


Overview

This project analyzes the Iris classification dataset using Python and Scikit-learn to build a
K-Nearest Neighbor (KNN) classifier. The project explores how feature scaling and distance
metrics drive model decisions, and identifies the optimal k value through cross-validation.

Dataset Used: Iris Dataset (150 samples, 4 features, 3 classes)

Project Goals:
- Load, clean, and scale numerical features for KNN
- Implement train/test splitting for unbiased evaluation
- Fit a KNN classifier and generate predictions on held-out test data
- Identify the optimal k value using cross-validation and performance plots
- Calculate and interpret accuracy, precision, recall, F1-score, and confusion matrix
- Explain how feature scaling and distance metrics impact KNN behavior.
  
Project Structure
- knn-classification/
- knn_classification.ipynb # Main Jupyter Notebook (all cells executed)
- README.md # Project description and instructions
- requirements.txt # Python dependencies
- 
Setup Instructions:
1. Clone the Repository
git clone https://github.com/elizabethdanladi/k-nearest-iris-analysis

2. Install Dependencies
pip install -r requirements.txt

4. Run the Notebook
Google Colab: Upload knn_classification.ipynb to colab.research.google.com
Jupyter Locally: Run jupyter notebook knn_classification.ipynb

Workflow Summary
Step Cell Description
1 Cell 1 Import all required libraries
2 Cell 2 Load Iris dataset and explore structure (150 rows, 5 columns, no nulls)
3 Cell 3 Check missing values (none found) and visualize with pairplot
4 Cell 4 Split data — 120 training / 30 testing samples (80/20)
5 Cell 5 Normalize features using StandardScaler
6 Cell 6 Cross-validate k values: 1, 3, 5, 7, 9, 11
7 Cell 7 Plot k vs. accuracy — Best K identified as k=3
8 Cell 8 Train final KNN model with k=3
9 Cell 9 Evaluate model performance
10 Cell 10 Test predictions on 3 new sample data points.

Key Findings
Cross-Validation Results
K Value CV Accuracy
1 0.9417
3 0.9500 Best
5 0.9250
7 0.9417
9 0.9417
11 0.9500

Optimal K = 3 (tied with k=11, but k=3 is preferred for simplicity and lower overfitting
risk)

Final Model Performance (k=3)
Metric and Score
Accuracy 1.0 (100%)
Precision 1.0 (100%)
Recall 1.0 (100%)
F1 Score 1.0 (100%)

Confusion Matrix
The model correctly classified all 30 test samples with zero misclassifications across all 3
Iris species (Setosa, Versicolor, Virginica).
Sample Predictions on New Data
Sample and Features and Predicted Class
1 [5.1, 3.5, 1.4, 0.2] Setosa
2 [6.7, 3.0, 5.2, 2.3] Virginica
3 [5.8, 2.7, 4.1, 1.0] Versicolor

Key Insights
Feature scaling is critical — KNN relies on Euclidean distance, so unscaled features
with larger ranges would dominate and bias results
k=3 was optimal — Small k values like 1 risk overfitting; k=3 balances bias and variance
well on this dataset
Perfect test accuracy (100%) is realistic for the Iris dataset, which is well-separated
and clean with no noise.

Dependencies
See requirements.txt for full list. Main libraries:
scikit-learn — KNN model, metrics, preprocessing
pandas — Data manipulation
numpy — Numerical operations
matplotlib / seaborn — Visualization

Author
Elizabeth Danladi Sabatu
