import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = load_iris()

print(data.data.shape)
print(data.target.shape)
x = pd.DataFrame(data.data, columns=data.feature_names)
print(x.head())
y = pd.DataFrame(data.target, columns=['target'])
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train.values.ravel())

import joblib
joblib.dump(model, 'model.joblib')