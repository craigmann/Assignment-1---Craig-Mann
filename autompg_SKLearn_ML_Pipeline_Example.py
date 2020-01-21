# Sci-Kit Learn Pipeline Example - Best Model
# Source - https://www.kdnuggets.com/2017/12/managing-machine-learning-workflows-scikit-learn-pipelines-part-1.html


import os
from settings import APP_ROOT
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import joblib
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn import tree
import pandas as pd

# Load and split the data
autompg = pd.read_csv(os.path.join(APP_ROOT, "Assignment-1---Craig-Mann\\auto-mpg.csv"), 
					header=0, na_values="?", comment="\t")

X = autompg.drop('mpg', axis=1)
y = autompg['mpg']

X_train, X_test, y_train, y_test = train_test_split(X, 
													y, 
													test_size=0.2,
													random_state=42)

# Create categorical and numerical transformers for pipeline.
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer(
	transformers=[
		('num', numeric_transformer,numeric_features),
		('cat', categorical_transformer, categorical_features)
	]
)

# Construct the pipelines

pipe_lr = Pipeline([('scl', StandardScaler()),
			('pca', PCA(n_components=3)),
			('clf', LinearRegression())])

pipe_svr = Pipeline([('scl', StandardScaler()),
			('pca', PCA(n_components=3)),
			('clf', svm.SVR())])
			
pipe_dt = Pipeline([('scl', StandardScaler()),
			('pca', PCA(n_components=3)),
			('clf', tree.DecisionTreeRegressor(random_state=42))])

# List of pipelines for ease of iteration
pipelines = [pipe_lr, pipe_svr, pipe_dt]
			
# Dictionary of pipelines and classifier types for ease of reference
pipe_dict = {0: 'Linear Regression', 1: 'Support Vector Machine', 2: 'Decision Tree'}

# Fit the pipelines
for pipe in pipelines:
	pipe.fit(X_train, y_train)

# Compare accuracies
for idx, val in enumerate(pipelines):
	print('%s pipeline test accuracy: %.3f' % (pipe_dict[idx], val.score(X_test, y_test)))

# Identify the most accurate model on test data
best_acc = 0.0
best_clf = 0
best_pipe = ''
for idx, val in enumerate(pipelines):
	if val.score(X_test, y_test) > best_acc:
		best_acc = val.score(X_test, y_test)
		best_pipe = val
		best_clf = idx
print('Classifier with best accuracy: %s' % pipe_dict[best_clf])

# Save pipeline to file
joblib.dump(best_pipe, 'best_pipeline.pkl', compress=1)
print('Saved %s pipeline to file' % pipe_dict[best_clf])

# Load Pipeline
load_best_pipe = joblib.load('best_pipeline.pkl')
y_test_new = load_best_pipe.predict(X_test)
print(y_test_new)
print(y_test)
