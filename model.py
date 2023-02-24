import pandas as pd

df = pd.read_csv("insurance.csv")
print(df.head())

df = df.drop('index', axis=1)

# Replacing string values to numbers
df['sex'] = df['sex'].apply({'male':0, 'female':1}.get)
df['smoker'] = df['smoker'].apply({'yes':1, 'no':0}.get)
df['region'] = df['region'].apply({'southwest':1, 'southeast':2, 'northwest':3, 'northeast':4}.get)

print(df.head())

# features
X = df[['age', 'sex', 'bmi', 'children','smoker','region']]
# predicted variable
y = df['charges']

# importing train_test_split model
from sklearn.model_selection import train_test_split

# splitting train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import KFold
def perform_cross_validation(model):
    kf = KFold(n_splits=5)
    cv_scores_train = []
    cv_scores_test = []
    for train_indices, test_indices in kf.split(X_train):
        train = X_train.iloc[train_indices, :]
        train_targets = y_train.iloc[train_indices]
        test = X_train.iloc[test_indices, :]
        test_targets = y_train.iloc[test_indices]

        model.fit(train, train_targets)
        cv_scores_train.append(model.score(train, train_targets))
        cv_scores_test.append(model.score(test, test_targets))

    print('Mean R2 score for train: ', sum(cv_scores_train) / 5)
    print('Mean R2 score for test: ', sum(cv_scores_test) / 5)

model_rf = RandomForestRegressor(n_estimators=10, max_depth=20, min_samples_leaf=5, min_samples_split=5, random_state=42)
perform_cross_validation(model_rf)

model_rf.fit(X_train, y_train)

import pickle

pickle.dump(model_rf, open("model.pkl", "wb"))