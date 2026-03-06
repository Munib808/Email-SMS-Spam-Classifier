import streamlit as st
import pickle
import string

# Load your TF-IDF vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Simple stemmer (Porter Stemmer alternative using suffix stripping)
def simple_stem(word):
    suffixes = ['ing', 'ly', 'ed', 'ious', 'ies', 'ive', 'es', 's', 'ment']
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[:-len(suffix)]
    return word

# Minimal stopwords list (common English words)
STOPWORDS = set([
    'a','an','the','is','it','this','that','in','on','for','to','of','and','or','if','are','as','at','be','by','from','has','he','she','i','you','we','they','them','with','was','but'
])

def transform_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize by whitespace
    words = text.split()
    # Remove stopwords
    words = [word for word in words if word not in STOPWORDS]
    # Apply simple stemming
    words = [simple_stem(word) for word in words]
    return ' '.join(words)

# Streamlit UI
st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter your message here:")

if st.button('Predict'):
    if input_sms.strip() == "":
        st.warning("Please enter a message to classify!")
    else:
        # Preprocess
        transformed_sms = transform_text(input_sms)
        # Vectorize
        vector_input = tfidf.transform([transformed_sms])
        # Predict
        result = model.predict(vector_input)[0]
        # Display result
        if result == 1:
            st.header("Spam 🚫")
        else:
            st.header("Not Spam ✅")
