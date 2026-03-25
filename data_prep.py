import pandas as pd
from sklearn.utils import shuffle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tqdm import tqdm
nltk.download('stopwords')
true_df=pd.read_csv('True.csv')
fake_df=pd.read_csv('Fake.csv')
true_df['label']=1
fake_df['label']=0
df=pd.concat([true_df,fake_df])
df=shuffle(df)
print(df.isnull().sum())
df=df.reset_index(drop=True)
df.dropna(inplace=True)
# print(df.head(10))
# data labelled hogya aur empty part hat gaya
# ab standarization karna hain matlab sabhi grammer ko lowercaase karna , punctuation aur numbers hatana ,
# stopwords -> the", "is", "at", "which", and "on" ko remove karna beacause wo sirf model training time badhate hain
stop_words = set(stopwords.words('english'))
regex_pattern = re.compile('[^a-zA-Z]')
port_stem=PorterStemmer()
def clean_text(content):
    stemmed_content = regex_pattern.sub(' ', str(content))
    stemmed_content=stemmed_content.lower()
    stemmed_content=stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stop_words]
    stemmed_content=' '.join(stemmed_content)
    return stemmed_content

def clean_text_dl(content):
    cleaned_content=regex_pattern.sub(' ', str(content))
    cleaned_content=cleaned_content.lower()
    return cleaned_content

tqdm.pandas(desc="cleaning text")
df['content']=df['title']+" "+df['text']
#neural cleaning
df['content_dl']=df['content'].progress_apply(clean_text_dl)
print("starting")
#logistic cleaning
df['content']=df['content'].progress_apply(clean_text)
df=df.drop(['title','text','subject','date'],axis=1)
df_dl=df.drop(['content'],axis=1)
df_dl=df_dl.rename(columns={'content_dl':'content'})
#axis =1  se column delete hoti hain and axis=0 se row
df_dl.to_csv('Cleaned_News_dl.csv',index=False)
df_logistic=df.drop(['content_dl'],axis=1)
df_logistic.to_csv('Cleaned_News.csv',index=False)
# saving our cleaned data into new csv file