from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn import datasets
bunch=datasets.load_breast_cancer()
X_train,X_test,Y_train,Y_test=train_test_split(bunch.data,bunch.target,test_size=0.3,shuffle=True,stratify=bunch.target)


svc=SVC(C=1e5,kernel="linear",
       gamma="scale",class_weight="balanced")

selector=RFE(estimator=svc,n_features_to_select=27)
selector=selector.fit(X_train,Y_train)
X_train=selector.transform(X_train)
X_test=selector.transform(X_test)

svc.fit(X_train,Y_train)
acc=svc.score(X_test,Y_test)
print("Accuracy",acc)
acc1=svc.score(X_train,Y_train)
print("Accuracy",acc1)