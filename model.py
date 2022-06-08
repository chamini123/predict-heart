# importing modules
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
# from matplotlib import pyplot as plt

# preprocessing data
data = pd.read_csv("heart.csv")
features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'restecg', 'thalach',
            'oldpeak', 'ca', 'thal']
X = data[features]
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                    random_state=1)

# comparing classification models

# KNN classification
k = range(1, 50)
scores = []
for i in k:     # evaluating model for different values of k
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))
# plt.plot(k, scores)
# plt.title('k vs accuracy')
# plt.xlabel('k value')
# plt.ylabel('accuracy score')
# plt.savefig('kvsScores.png')
# plt.show()

# fitting the model with best hyperparameter
knn = KNeighborsClassifier(n_neighbors=32)
knn.fit(X, y)


# Logistic Regression
logreg = LogisticRegression(solver='newton-cg')
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
# print(metrics.accuracy_score(y_test, y_pred))

logreg = LogisticRegression(solver='liblinear')
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
# print(metrics.accuracy_score(y_test, y_pred))

# fitting the model with best hyperparameter
logreg = LogisticRegression(solver='newton-cg')
logreg.fit(X, y)


# based on accuracy, considering the best Model
Model = logreg      # logistic regression (newton-cg) has highest accuracy
