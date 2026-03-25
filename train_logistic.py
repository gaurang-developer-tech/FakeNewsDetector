import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,cross_val_score,StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score,precision_score,ConfusionMatrixDisplay,RocCurveDisplay,classification_report

#loading the data from file label and tfidf_matrix
X=joblib.load('tfidf_matrix_X.joblib')
Y=joblib.load('label_Y.joblib')

# Set up the Model and K-Fold Configuration
model=LogisticRegression(max_iter=1000 )
kf=StratifiedKFold(n_splits=10,shuffle=True,random_state=42)

print("running 10-fold cross validation")

cv_score=cross_val_score(model,X,Y,cv=kf,scoring='accuracy')
print(f"accuracy score is: {cv_score} ")
print(f"Average robust Accuracy score is: {np.mean(cv_score)}")

#training our final model once
print("training our final model and generating graph accordingly")
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
model.fit(X_train,Y_train)
Y_pred=model.predict(X_test)

print("the final classification report is:")
print(classification_report(Y_test,Y_pred))

print("Generating performance graph")
fig,ax=plt.subplots(1,2,figsize=(10,5))

#left slot containing confusion matrix graph
ConfusionMatrixDisplay.from_estimator(model,X_test,Y_test,ax=ax[0],cmap='Blues',display_labels=['Fake','Real'])
ax[0].set_title('Confusion Matrix')

#right slot containing ROC
RocCurveDisplay.from_estimator(model,X_test,Y_test,ax=ax[1])
ax[1].set_title('ROC Curve')

plt.tight_layout()
plt.show()

print("Now saving the trained model")
joblib.dump(model,'logistic_model.joblib')
print("Model saved successfully")