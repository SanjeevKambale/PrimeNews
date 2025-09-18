import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load data
true_data = pd.read_csv(r'C:\MyProject\PrimeNews\Dataset\True.csv')
fake_data = pd.read_csv(r'C:\MyProject\PrimeNews\Dataset\Fake.csv')
true_data['label'] = 1
fake_data['label'] = 0
news = pd.concat([true_data, fake_data], axis=0)
news = news.sample(frac=1).reset_index(drop=True)
news = news.drop(['subject', 'date', 'title'], axis=1)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\n', ' ', text)
    return text

news['text'] = news['text'].apply(clean_text)
x = news['text']
y = news['label']

vectorizer = TfidfVectorizer(max_features=5000)
xv = vectorizer.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(xv, y, test_size=0.3)

model_lr = LogisticRegression()
model_lr.fit(x_train, y_train)

model_dtc = DecisionTreeClassifier()
model_dtc.fit(x_train, y_train)

# Save models and vectorizer
joblib.dump(vectorizer, r'C:\MyProject\PrimeNews\Models\vectorizer.joblib')
joblib.dump(model_lr, r'C:\MyProject\PrimeNews\Models\lr_model.joblib')
joblib.dump(model_dtc, r'C:\MyProject\PrimeNews\Models\dtc_model.joblib')

print('Models and vectorizer saved successfully!')
