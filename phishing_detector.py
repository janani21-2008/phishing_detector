
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt


emails = [

    "Meeting scheduled tomorrow",
    "Project report attached",
    "Lunch meeting confirmed",
    "Happy birthday have a great day",
    "Your electricity bill has been paid",
    "Team meeting at 5 PM",

    "Click here to win free money",
    "Your bank account is suspended",
    "Verify your password immediately",
    "Claim your free iPhone now",
    "Update your payment information",
    "Urgent login to your account"

]

labels = [
    0,0,0,0,0,0,
    1,1,1,1,1,1
]

df = pd.DataFrame({
    "email": emails,
    "label": labels
})


X = df["email"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)


model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy Score:")
print(accuracy)


cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


print("\nClassification Report:")
print(classification_report(y_test, y_pred))


plt.imshow(cm, cmap="Blues")

plt.title("Confusion Matrix")

plt.xticks([0,1], ["Safe", "Phishing"])
plt.yticks([0,1], ["Safe", "Phishing"])

plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j],
                 ha="center", va="center")

plt.colorbar()
plt.show()


test_email = [
    "Click here immediately to verify your bank account"
]

prediction = model.predict(test_email)

if prediction[0] == 1:
    print("\nResult: Phishing Email")
else:
    print("\nResult: Safe Email")