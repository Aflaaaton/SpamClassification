import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


def text_transformer(text):
    text = text.lower()  # Creating lower Case
    text = nltk.word_tokenize(text)  # Tokenizing the text

    y = []
    for i in text:
        if i.isalnum():  # Removing Special Characters
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:  # Removing Stopwords and Punctuation
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))  # Stemming

    return ' '.join(y)