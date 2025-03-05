# disease_prediction_system

## Machine Learning Models
Random Forest (Ensemble, robust, low bias)
SVM (Max-margin, kernel trick)
KNN (Distance-based, slow for large data)
Logistic Regression (Probabilistic, interpretable)
MLP Classifier (Deep learning, backpropagation)
XGBoost (Boosting, high accuracy, prevents overfitting)

## Strengths & Limitations
Random Forest & XGBoost: High accuracy, handles missing data but computationally expensive.
SVM & MLP: Powerful but require tuning and resources.
KNN & Decision Trees: Simple & was highly accurate in this case but inefficient for large datasets.

## XGBoost as Meta-Learner
Combines strengths of base models.
Prevents overfitting, fast, handles missing data well.

## Methodology & Model Training
Preprocessing: Missing data handling, feature encoding, scaling.
Training: Base models → Predictions → XGBoost meta-learner.
Evaluation: Accuracy, precision, recall, F1-score.

## Challenges & Solutions
Overfitting: Solved using cross-validation, regularization.
Hyperparameter tuning: Optimized for best performance.
Data Issues: Missing values handled via imputation.

## Future Improvements
Integrate medical imaging, real-time doctor consultations.


## UI
**Landing Page**
![image](https://github.com/user-attachments/assets/c4107f46-4a49-483c-bed6-d94a6300c665)
![image](https://github.com/user-attachments/assets/f3f6aff4-4903-453e-965a-77835da4c5b6)

**Registration Page**
![image](https://github.com/user-attachments/assets/8687a540-6c9c-4a78-81c3-0882ce05e40d)

**Symptom Selection Page**
![image](https://github.com/user-attachments/assets/c1ffc574-2795-4d96-8143-36b3859d7c61)
![image](https://github.com/user-attachments/assets/b812d09a-1227-4764-80ac-c1f56ed179af)

**Results Page**
![image](https://github.com/user-attachments/assets/03370e6b-8fc5-4b04-a128-48d596fc1357)

