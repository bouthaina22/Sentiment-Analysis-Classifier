import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



data = pd.read_csv("data.csv")
print(data)
X = data["text"]
y = data["sentiment"]

vectorizer = TfidfVectorizer()
#X_numbers = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

#vectorizer = TfidfVectorizer()
vectorizer = TfidfVectorizer(ngram_range=(1, 2))

X_train_numbers = vectorizer.fit_transform(X_train)
X_test_numbers = vectorizer.transform(X_test)
#model = LogisticRegression()
model = LogisticRegression(class_weight="balanced")
model.fit(X_train_numbers, y_train)
predictions = model.predict(X_test_numbers)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
print(classification_report(y_test, predictions))
for text, real, predicted in zip(X_test, y_test, predictions):
    print(text, "| Real:", real, "| Predicted:", predicted)

#print(X_numbers.toarray())

#from sklearn.linear_model import LogisticRegression
#model = LogisticRegression()
#model.fit(X_numbers, y)
#print("Model trained successfully!")


cm = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(cm)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)
disp.plot()


test_sentences = [
    "I love this app",
    "I hate this app",
    "This app is okay",
    "I don't like this application",
    "This application is fantastic"
]

for sentence in test_sentences:
    sentence_numbers = vectorizer.transform([sentence])
    prediction = model.predict(sentence_numbers)
    print(sentence, "->", prediction[0])





#new_text = ["I really love this app"]
new_text = input("Enter a sentence: ")
new_text_numbers = vectorizer.transform([new_text])
prediction = model.predict(new_text_numbers)

print("Prediction:", prediction[0])



#
#print(X)
#print(y)