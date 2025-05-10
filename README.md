
# 📧 Email Spam Detector

A machine learning-based Email Spam Detector that classifies incoming emails as **"Spam"** or **"Not Spam"**, automating the process of filtering unwanted messages. This project includes model training, performance evaluation, and a deployable user interface using Streamlit.

---

## 🚀 Features

- 🔍 **Preprocessing**: Cleans and tokenizes raw email text.
- 🧠 **Machine Learning**: Trained classification model using feature extraction (e.g., TF-IDF).
- 📊 **Evaluation**: Reports performance using accuracy, precision, recall, and F1-score.
- 📦 **Model Export**: Saves the trained model as a `.pkl` file.
- 🌐 **Streamlit App**: User-friendly interface for real-time email classification.
- ☁️ **Deployment**: Hosted on Streamlit Community Cloud.

---

## 🎯 Objectives

- Automate the classification of emails into **Spam** or **Not Spam**.
- Improve detection accuracy through model training and evaluation.
- Integrate the classifier with email systems for seamless filtering.

---

## 📁 Project Structure

```
email-spam-detector/
│
├── data/
│   └── emails.csv                  # Cleaned and labeled dataset
│
├── models/
│   └── spam_classifier.pkl         # Exported trained model
│
├── app/
│   └── streamlit_app.py            # Streamlit web interface
│
├── src/
│   ├── preprocess.py               # Text preprocessing functions
│   ├── feature_extraction.py       # TF-IDF or CountVectorizer logic
│   └── model_training.py           # ML model training and evaluation
│
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
└── spam_detector.py                # Email classification script
```

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/your-username/email-spam-detector.git
cd email-spam-detector

# Install required libraries
pip install -r requirements.txt
```

---

## 📊 Model Training

Train the model using the preprocessed dataset:

```bash
python src/model_training.py
```

This script will:
- Load and preprocess the data
- Extract features using TF-IDF
- Train a classifier (e.g., Naive Bayes or SVM)
- Evaluate performance
- Save the model as `spam_classifier.pkl`

---

## 🧪 Evaluation Metrics

The model is evaluated using:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **Confusion Matrix**

These metrics are displayed after training and stored in a performance report.

---

## 🖥️ Streamlit App

To launch the web app locally:

```bash
streamlit run app/streamlit_app.py
```

Enter or paste email text, and the app will display the classification result (Spam/Not Spam).

---

## ☁️ Deployment

To deploy on [Streamlit Community Cloud](https://streamlit.io/cloud):
1. Push your project to GitHub.
2. Go to the Streamlit Community Cloud and connect your GitHub repository.
3. Set the main file as `app/streamlit_app.py`.
4. Click **Deploy**.

---

## ✅ Requirements

- Python 3.7+
- Scikit-learn
- Pandas
- Numpy
- Streamlit
- NLTK / spaCy (for preprocessing)

---

## 📦 Model Export

The trained model is exported as:

```
models/spam_classifier.pkl
```

You can load it using:

```python
import joblib
model = joblib.load('models/spam_classifier.pkl')
```

---

## 📬 Integration

To integrate this with an email system (like Outlook or Gmail via APIs), use the `spam_detector.py` script to classify incoming emails and move them to folders accordingly.

---

## 📜 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

Developed by **[Your Name]**  
[GitHub](https://github.com/your-username) | [LinkedIn](https://linkedin.com/in/your-link)
