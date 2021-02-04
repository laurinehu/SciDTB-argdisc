import nltk
import re
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

stemmer = SnowballStemmer("english")

def preprocess(text):
    # remove stopwords
    sw = nltk.corpus.stopwords.words('english')
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    tokens = [tok.lower() for tok in tokens]
    tokens = [tok for tok in tokens if tok not in sw]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

def vectorize(texts):
    #define vectorizer parameters
    tfidf_vectorizer = TfidfVectorizer(tokenizer=preprocess)
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts) #fit the vectorizer to synopses
    return tfidf_matrix
