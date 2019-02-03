import nltk
import pickle
import re
import numpy as np
from gensim.models import KeyedVectors
import pandas as pd

nltk.download('stopwords')
from nltk.corpus import stopwords

# Paths for all resources for the bot.
RESOURCE_PATH = {
    'INTENT_RECOGNIZER': 'intent_recognizer.pkl',
    'TAG_CLASSIFIER': 'tag_classifier.pkl',
    'TFIDF_VECTORIZER': 'tfidf_vectorizer.pkl',
    'THREAD_EMBEDDINGS_FOLDER': 'thread_embeddings_by_tags',
    'WORD_EMBEDDINGS': 'word_embeddings.tsv',
}


def text_prepare(text):
    """Performs tokenization and simple preprocessing."""
    
    replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
    bad_symbols_re = re.compile('[^0-9a-z #+_]')
    stopwords_set = set(stopwords.words('english'))

    text = text.lower()
    text = replace_by_space_re.sub(' ', text)
    text = bad_symbols_re.sub('', text)
    text = ' '.join([x for x in text.split() if x and x not in stopwords_set])

    return [text.strip()]


def load_embeddings(embeddings_path, binary = False):
    """Loads pre-trained word embeddings from tsv file.

    Args:
      embeddings_path - path to the embeddings file.

    Returns:
      embeddings - dict mapping words to vectors;
      embeddings_dim - dimension of the vectors.
    """
    
    # Hint: you have already implemented a similar routine in the 3rd assignment.
    # Note that here you also need to know the dimension of the loaded embeddings.
    # When you load the embeddings, use numpy.float32 type as dtype

    ########################
    #### YOUR CODE HERE ####
    ########################
    wv_embeddings = KeyedVectors.load_word2vec_format(embeddings_path, datatype=np.float32, binary=binary)
    
    return wv_embeddings, wv_embeddings.vector_size

def question_to_vec(question, embeddings, dim):
    """Transforms a string to an embedding by averaging word embeddings."""
    
    # Hint: you have already implemented exactly this function in the 3rd assignment.

    ########################
    #### YOUR CODE HERE ####
    ########################
    c = 0 
    result = []
    question_tokens = question.split(" ")
        
    for word in question_tokens:
        if word in embeddings:
            result.append(embeddings[word])
            c+=1
            
    if c == 0:
        return np.zeros((c+1,dim))
    else:
        return np.array(result, dtype=np.float32).mean(axis = 0)


def unpickle_file(filename):
    """Returns the result of unpickling the file content."""
    # with open(filename, 'rb') as f:
    #    return pickle.load(f)
    return pd.read_pickle(filename)
