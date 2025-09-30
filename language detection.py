import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Load the dataset
data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/dataset.csv")


# Display first few rows
print(data.head())
# Check for null values
print("Null values in each column:\n", data.isnull().sum())

# Display language counts
print("\nLanguage distribution:\n", data["language"].value_counts())

# Split data into features and labels
x = np.array(data["Text"])
y = np.array(data["language"])

# Text vectorization
cv = CountVectorizer()
X = cv.fit_transform(x)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate model
accuracy = model.score(X_test, y_test)
print("\nModel Accuracy:", accuracy)

# Predict language for user input
user_input = input("\nEnter a Text: ")
user_data = cv.transform([user_input]).toarray()
output = model.predict(user_data)

print("Predicted Language:", output[0])
