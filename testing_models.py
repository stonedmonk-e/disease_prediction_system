import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier
from warnings import simplefilter
from sklearn.metrics import precision_score
simplefilter(action='ignore', category=FutureWarning)

# Load train data 
df_train = pd.read_csv('Data/Training.csv')

# Split features and target for train data
cols = df_train.columns
cols = cols[:-2] # Last Column is dropped as it is an extra columns read by pandas library with NaN values
train_features = df_train[cols]
train_labels = df_train['prognosis']

# Load test data 
df_test = pd.read_csv('Data/Testing.csv')

# Split features and target for test data
cols = df_test.columns
cols = cols[:-1] # Last Column is dropped as it is an extra columns read by pandas library with NaN values
test_features = df_test[cols]
test_labels = df_test['prognosis']

# Check for data sanity
#assert (len(train_features.iloc[0]) == 132)
#assert (len(train_labels) == train_features.shape[0])

print("Length of Training Data: ", df_train.shape)

print("Training Features: ", train_features.shape)
print("Training Labels: ", train_labels.shape)



# Train Random forest model
m1_rf = RandomForestClassifier()
m1_rf.fit(train_features, train_labels) 
# Evaluate on test data
print("Random Forrest Classifier Accuracy: ",m1_rf.score(test_features, test_labels))

# Train Decision Tree model
m2_dt = DecisionTreeClassifier()
m2_dt.fit(train_features, train_labels) 
# Evaluate on test data
print("Decision Tree Accuracy: ",m2_dt.score(test_features, test_labels))

# Train SVM model
m3_svm = SVC()
m3_svm.fit(train_features, train_labels) 
# Evaluate on test data
print("Support Vector Machine Accuracy: ",m3_svm.score(test_features, test_labels))

# Train KNN model
m4_knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski')
m4_knn.fit(train_features, train_labels) 
# Evaluate on test data
print("K Nearest Neighbours Accuracy: ",m4_knn.score(test_features, test_labels))

# Train Logistic Regression model
m5_lr =  LogisticRegression()
m5_lr.fit(train_features, train_labels) 
# Evaluate on test data
print("Logistic Regression Accuracy: ",m5_lr.score(test_features, test_labels))

# Train MLP model
m6_mlp = MLPClassifier()
m6_mlp.fit(train_features, train_labels) 
# Evaluate on test data
print("MLP Classifier Accuracy: ",m6_mlp.score(test_features, test_labels))

# Train Gradient Boost 
m7_gb = GradientBoostingClassifier()
m7_gb.fit(train_features, train_labels)
#Evaluate on test data
print("Gradient Boost Classifier Accuracy: ",m7_gb.score(test_features, test_labels))

estimator_list = [('knn',m4_knn),('svm',m3_svm),('lr',m5_lr), ('rf',m1_rf),('mlp',m6_mlp),]
# Build stack model
stack_model = StackingClassifier(
    estimators=estimator_list, final_estimator=XGBClassifier()
)

# Train stacked model
stack_model.fit(train_features, train_labels)
#Evaluate Stacking Model
print("XGBoost Stacked Model Accuracy:",stack_model.score(test_features,test_labels))
"""
print("Please enter values for the following symptoms:")
input_data = {}
for feature in train_features.columns:
    val = input(f"{feature}: ")
    input_data[feature] = [float(val)]

# Convert input into DataFrame
input_df = pd.DataFrame(input_data)

# Make predictions for each model
prediction_rf = m1_rf.predict(input_df)
prediction_dt = m2_dt.predict(input_df)
prediction_svm = m3_svm.predict(input_df)
prediction_knn = m4_knn.predict(input_df)
prediction_lr = m5_lr.predict(input_df)
prediction_mlp = m6_mlp.predict(input_df)
prediction_gb = m7_gb.predict(input_df)

# Print predictions
print("Random Forest Prediction:", prediction_rf[0])
print("Decision Tree Prediction:", prediction_dt[0])
print("Support Vector Machine Prediction:", prediction_svm[0])
print("K-Nearest Neighbors Prediction:", prediction_knn[0])
print("Logistic Regression Prediction:", prediction_lr[0])
print("MLP Prediction:", prediction_mlp[0])
print("Gradient Boost Prediction:", prediction_gb[0])
"""