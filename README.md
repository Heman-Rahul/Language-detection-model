# 🌍 Language Detection using Naive Bayes

This project is a **Language Detection Model** built with Python and Scikit-learn.
It predicts the language of a given text input using a **Multinomial Naive Bayes classifier** trained on a multilingual dataset.

---

## 📌 Features

* Detects language of input text.
* Uses **Bag of Words (CountVectorizer)** for text vectorization.
* Implements **Multinomial Naive Bayes** for classification.
* Achieves good accuracy on the test dataset.
* Interactive: users can input text and get predicted language instantly.

---

## ⚙️ Tech Stack

* **Python 3**
* **Pandas & NumPy** → Data handling
* **Scikit-learn** → Text vectorization + Model training
* **Naive Bayes Classifier** → Language prediction

---

## 📂 Dataset

The dataset used is from [Aman Kharwal’s GitHub](https://github.com/amankharwal/Website-data/blob/master/dataset.csv).
It contains text samples in different languages along with labels.



```
Model Accuracy: 0.96
```

---

## 📌 Future Improvements

* Use **TF-IDF Vectorizer** for better feature extraction.
* Train with a larger multilingual dataset.
* Deploy as a **Flask/Django web app** or **Streamlit app** for user-friendly UI.
