{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "f82482fe",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2023-09-08 00:27:05.990 INFO    numexpr.utils: Note: NumExpr detected 16 cores but \"NUMEXPR_MAX_THREADS\" not set, so enforcing safe limit of 8.\n",
      "2023-09-08 00:27:06.013 INFO    numexpr.utils: NumExpr defaulting to 8 threads.\n",
      "2023-09-08 00:27:10.841 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run c:\\pythonn\\lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import pickle\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from PIL import Image\n",
    "import re\n",
    "import string\n",
    "import nltk\n",
    "import spacy\n",
    "\n",
    "with open(\"svm_model.pkl\", \"rb\") as file:\n",
    "    model = pickle.load(file)\n",
    "\n",
    "with open(\"tfidf_vectorizer.pkl\", \"rb\") as file:\n",
    "    vectorizer = pickle.load(file)\n",
    "\n",
    "nltk.download('stopwords')\n",
    "stopwords = nltk.corpus.stopwords.words('english')\n",
    "\n",
    "def clean_text(text):\n",
    "    text = text.lower()\n",
    "    return text.strip()\n",
    "\n",
    "def remove_punctuation(text):\n",
    "    punctuation_free = \"\".join([i for i in text if i not in string.punctuation])\n",
    "    return punctuation_free\n",
    "\n",
    "def tokenization(text):\n",
    "    tokens = re.split(' ', text)\n",
    "    return tokens\n",
    "\n",
    "def remove_stopwords(text):\n",
    "    output = \" \".join(i for i in text if i not in stopwords)\n",
    "    return output\n",
    "\n",
    "def lemmatizer(text):\n",
    "    nlp = spacy.load('en_core_web_sm')\n",
    "    doc = nlp(text)\n",
    "    sent = [token.lemma_ for token in doc if not token.text in set(stopwords)]\n",
    "    return ' '.join(sent)\n",
    "\n",
    "st.title(\"Sentiment Analysis App\")\n",
    "st.markdown(\"By Nirmal Gaud\")\n",
    "image = Image.open(\"sentiment.png\")\n",
    "st.image(image, use_column_width=True)\n",
    "\n",
    "st.subheader(\"Enter your text here:\")\n",
    "user_input = st.text_area(\"\")\n",
    "\n",
    "if user_input:\n",
    "    user_input = clean_text(user_input)\n",
    "    user_input = remove_punctuation(user_input)\n",
    "    user_input = user_input.lower()\n",
    "    user_input = tokenization(user_input)\n",
    "    user_input = remove_stopwords(user_input)\n",
    "    user_input = lemmatizer(user_input)\n",
    "\n",
    "if st.button(\"Predict\"):\n",
    "    if user_input:\n",
    "        text_vectorized = vectorizer.transform([user_input])\n",
    "        prediction = model.predict(text_vectorized)[0]\n",
    "        st.header(\"Prediction:\")\n",
    "        if prediction == -1:\n",
    "            st.subheader(\"The sentiment of the given text is: Negative\")\n",
    "        elif prediction == 0:\n",
    "            st.subheader(\"The sentiment of the given text is: Neutral\")\n",
    "        elif prediction == 1:\n",
    "            st.subheader(\"The sentiment of the given text is: Positive\")\n",
    "    else:\n",
    "        st.subheader(\"Please enter a text for prediction.\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
