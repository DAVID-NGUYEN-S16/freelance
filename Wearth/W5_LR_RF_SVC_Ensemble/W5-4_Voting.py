
from sklearn import datasets
bunch = datasets.load_breast_cancer()

data = bunch.data
labels = bunch.target


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(data, labels, test_size=0.3,shuffle=True,stratify=labels)



from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

clf1 = LogisticRegression(C=1e5)
clf2 = RandomForestClassifier(n_estimators=50, class_weight='balanced')
clf3 = SVC(C=1e5,kernel='linear', gamma='scale', class_weight='balanced')

from sklearn.ensemble import VotingClassifier
votingClassifier = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')

votingClassifier.fit(X_train, Y_train)

acc = votingClassifier.score(X_test, Y_test)

print('Accuracy:',acc)
