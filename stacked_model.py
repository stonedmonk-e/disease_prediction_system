import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier
import xgboost as xgb
from joblib import dump
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

# Load train data 
df_train = pd.read_csv('Data\Training.csv')
# Split features and target for train data
cols = df_train.columns
cols = cols[:-2] # Last Column is dropped as it is an extra columns read by pandas library with NaN values
train_features = df_train[cols]
train_labels = df_train['prognosis']

# Load test data 
df_test = pd.read_csv('Data\Testing.csv')
# Split features and target for test data
cols = df_test.columns
cols = cols[:-1] # Last Column is dropped as it is an extra columns read by pandas library with NaN values
test_features = df_test[cols]
test_labels = df_test['prognosis']


# Train Random forest model
rf = RandomForestClassifier()
rf.fit(train_features, train_labels)
dump(rf, 'Model/rf.joblib')
print("Model-1 Done")


# Train SVM model
svm = SVC()
svm.fit(train_features, train_labels) 
dump(svm, 'Model/svm.joblib')
print("Model-2 Done")

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski')
knn.fit(train_features, train_labels) 
dump(knn, 'Model/knn.joblib')
print("Model-3 Done")

# Train Logistic Regression model
lr =  LogisticRegression()
lr.fit(train_features, train_labels)
dump(lr, 'Model/lr.joblib')
print("Model-4 Done")

# Train MLP model
mlp = MLPClassifier()
mlp.fit(train_features, train_labels) 
dump(mlp, 'Model/mlp.joblib')
print("Model-5 Done")

estimator_list = [
    ('knn',knn),
    ('svm',svm),
    ('lr',lr),
    ('rf',rf),
    ('mlp',mlp),]

# Build stack model
stack_model = StackingClassifier(
    estimators=estimator_list, final_estimator=xgb.XGBClassifier()
)

# Train stacked model
stack_model.fit(train_features, train_labels)
print("Stack Model Training Done")

dump(stack_model, 'Model/stacked_classifier.joblib')
print("Stack model has been saved to Model/stacked_classifier.joblib")

print("XGBoost Stacked Model Accuracy:",stack_model.score(test_features,test_labels))