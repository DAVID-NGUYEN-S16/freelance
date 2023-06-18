
from sklearn import datasets
bunch = datasets.load_breast_cancer()

data = bunch.data
labels = bunch.target

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(data, labels, test_size=0.3,shuffle=True,stratify=labels)


from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
baggingClassifier = BaggingClassifier(base_estimator=SVC(), n_estimators=10)
baggingClassifier.fit(X_train, Y_train)

predictions = baggingClassifier.predict(X_test)
accuracy = baggingClassifier.score(X_test, Y_test)

print(predictions)
print(accuracy)
