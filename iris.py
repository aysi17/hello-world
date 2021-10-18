from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection  import train_test_split
import matplotlib.pyplot as plt
flower= load_iris()
X_train,X_test,y_train,y_test=train_test_split(flower.data,flower.target,test_size=0.5)
model=LogisticRegression()
model.fit(X_train,y_train)
print(y_test[0])
model.predict(X_test)
print(model.score(X_test, y_test),y_test[0])
print(model.predict([X_test[0]]))
import joblib
joblib.dump(model,'irismodel')
model=joblib.load('irismodel')
print(model.predict([X_test[0]]))

