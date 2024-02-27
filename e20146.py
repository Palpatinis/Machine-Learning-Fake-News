import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
import re
from sklearn.pipeline import Pipeline
import time
#csv_file_path = r'' insert fake news 
data = pd.read_excel(csv_file_path)
data = data.dropna()
X = data['text']
def remove_hyperlinks(text):
    hyperlink_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    quotation_marks = re.compile(r'"([^"]+)"')
    citation =re.compile(r'\(([^)]+)\)')
    text_without_hyperlinks = re.sub(hyperlink_pattern, '', text)
    text_without_hyperlinks = re.sub(quotation_marks, '', text_without_hyperlinks)
    text_without_hyperlinks = re.sub(citation, '', text_without_hyperlinks)
    
    return text_without_hyperlinks
text_data_without_hyperlinks = [remove_hyperlinks(text) for text in X]

additional_stop_words = ['by', 'source', 'anonymous', 'researcher','state','confirmed','verified','evidence','suggests','according','records','according','reported','confirm','viewpoints','transparent','disclosure','journalistic','integrity']


default_stop_words = set(TfidfVectorizer(stop_words='english').get_stop_words())


combined_stop_words = list(default_stop_words.union(additional_stop_words))


y = data['label']
start_time1 = time.time()
print("Text results")
X_train, X_test, y_train, y_test = train_test_split(text_data_without_hyperlinks, y, test_size=0.1, random_state=42)
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000, stop_words=combined_stop_words, max_df=0.1)),
    ('model', LogisticRegression())
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Accuracy:", accuracy)
end_time1 = time.time()
execution_time = end_time1 - start_time1
print("Execution Time (seconds):", execution_time)
start_time1 = time.time()
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000, stop_words=combined_stop_words, max_df=0.7)),
    ('model', SVC(kernel='linear'))
])


pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Accuracy:", accuracy)
end_time1 = time.time()
execution_time = end_time1 - start_time1
print("Execution Time (seconds):", execution_time)






