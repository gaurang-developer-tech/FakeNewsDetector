import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
df=pd.read_csv('Cleaned_News.csv')
df=df.dropna(subset=['content'])
vectorizer=TfidfVectorizer(max_features=5000)
X=vectorizer.fit_transform(df['content'])
Y=df['label']
print ("saving features")
joblib.dump(X,'tfidf_matrix_X.joblib')
joblib.dump(Y,'label_Y.joblib')
joblib.dump(vectorizer,'tfidf_vectorizer.joblib')

print("Feature engineering done , data is ready for ml models")