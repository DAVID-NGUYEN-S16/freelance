from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn import datasets

bunch=datasets.load_breast_cancer()
label_names = bunch.target_names
print(label_names)
X_train,X_test,Y_train,Y_test=train_test_split(bunch.data,bunch.target,test_size=0.3,shuffle=True,stratify=bunch.target)
logreg=LogisticRegression(C=1e5,class_weight='balanced')
selector=RFE(estimator=logreg,n_features_to_select=27)
selector=selector.fit(X_train,Y_train)
X_train=selector.transform(X_train)
X_test=selector.transform(X_test)
a=logreg.fit(X_train,Y_train)
acc=logreg.score(X_test,Y_test)
print("Accuracy",acc)
acc1=logreg.score(X_train,Y_train)
print("Accuracy",acc1)