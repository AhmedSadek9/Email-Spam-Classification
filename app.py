# import streamlit as st
# import pickle
# import string
# from nltk.corpus import stopwords
# import nltk
# from nltk.stem.porter import PorterStemmer

# ps = PorterStemmer()


# def transform_text(text):
#     text = text.lower()
#     text = nltk.word_tokenize(text)

#     y = []
#     for i in text:
#         if i.isalnum():
#             y.append(i)

#     text = y[:]
#     y.clear()

#     for i in text:
#         if i not in stopwords.words('english') and i not in string.punctuation:
#             y.append(i)

#     text = y[:]
#     y.clear()

#     for i in text:
#         y.append(ps.stem(i))

#     return " ".join(y)

# tfidf = pickle.load(open('vectorizer.pkl','rb'))
# model = pickle.load(open('model.pkl','rb'))

# st.title("Email/SMS Spam Classifier")

# input_sms = st.text_area("Enter the message")

# if st.button('Predict'):

#     # 1. preprocess
#     transformed_sms = transform_text(input_sms)
#     # 2. vectorize
#     vector_input = tfidf.transform([transformed_sms])
#     # 3. predict
#     result = model.predict(vector_input)[0]
#     # 4. Display
#     if result == 1:
#         st.header("Spam")
#     else:
#         st.header("Not Spam")

##streamlit run app.py

## (spam)  Congratulations! You've won a $1000 Walmart gift card. Click the link below to claim your prize now. Offer valid only for today. http://spamlink.fakewin.com
## (spam) URGENT: Your account has been suspended. To reactivate, please verify your information immediately at http://verify-account-login.com. Failure to do so will result in permanent deactivation.
## (spam) You have won a free trip to Hawaii! Click the link below to claim your

##(Not Spam)  Hi John, just a reminder that our meeting is scheduled for 10 AM tomorrow. Let me know if you need to reschedule.


import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# Download NLTK data (if not already downloaded)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('punkt')
    nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load models
@st.cache_resource
def load_models():
    tfidf = pickle.load(open('vectorizer.pkl','rb'))
    model = pickle.load(open('model.pkl','rb'))
    return tfidf, model

tfidf, model = load_models()

# Custom CSS for styling
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .stTextArea textarea {
            height: 200px;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            padding: 10px 24px;
            font-weight: bold;
            width: 100%;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        .header {
            color: #2c3e50;
            text-align: center;
            padding-bottom: 20px;
        }
        .result {
            text-align: center;
            font-size: 24px;
            padding: 20px;
            border-radius: 5px;
            margin-top: 20px;
        }
        .spam {
            background-color: #ffdddd;
            color: #d63031;
            border: 1px solid #d63031;
        }
        .not-spam {
            background-color: #ddffdd;
            color: #27ae60;
            border: 1px solid #27ae60;
        }
        .footer {
            text-align: center;
            margin-top: 40px;
            color: #7f8c8d;
            font-size: 12px;
        }
    </style>
""", unsafe_allow_html=True)

# App layout
st.markdown('<h1 class="header">📧 Email/SMS Spam Classifier</h1>', unsafe_allow_html=True)

with st.expander("ℹ️ About this app"):
    st.write("""
    This app uses machine learning to classify messages as spam or not spam. 
    Paste your email or SMS message in the text area below and click the 'Predict' button to check.
    """)

col1, col2 = st.columns([3, 1])
with col1:
    input_sms = st.text_area("Enter your message here:", placeholder="Paste your email or SMS message here...")
with col2:
    st.markdown("### Tips:")
    st.markdown("- Check for suspicious links")
    st.markdown("- Look for urgent language")
    st.markdown("- Verify sender information")

if st.button('Predict', key='predict'):
    if not input_sms.strip():
        st.warning("Please enter a message to analyze.")
    else:
        with st.spinner('Analyzing message...'):
            # 1. preprocess
            transformed_sms = transform_text(input_sms)
            # 2. vectorize
            vector_input = tfidf.transform([transformed_sms])
            # 3. predict
            result = model.predict(vector_input)[0]
            
            
            
            if result == 1:
                st.markdown('<div class="result spam">⚠️ This message is SPAM</div>', unsafe_allow_html=True)
                st.markdown("### Characteristics of this spam message:")
                st.markdown("- " + "\n- ".join(transformed_sms.split()[:5]) + "...")
            else:
                st.markdown('<div class="result not-spam">✓ This message is NOT SPAM</div>', unsafe_allow_html=True)
                st.success("This message appears to be legitimate.")
                # Display results with animation
                st.balloons()

st.markdown('<div class="footer">Spam Classifier App | Made with Streamlit</div>', unsafe_allow_html=True)