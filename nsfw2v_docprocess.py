from nltk.corpus import stopwords
import string
import numpy as np


class document_vector:
    def __init__(self, model):
        self.model = model

    def preprocess(self, x: str):
        # create stoplist and puncutation list that includes everything except hyphen
        stoplist = stopwords.words('english')
        my_punc = string.punctuation.replace('-', '')
        # standard preprocessing
        x = x.lower()
        x = [w for w in x.split() if w not in stoplist and not w.isdigit() and len(w) > 1]
        # remove punctuation from words except for hyphen
        x = " ".join(word for word in x if word not in stoplist)
        x = x.translate(str.maketrans('', '', my_punc))
        x = [w for w in x.split() if w not in stoplist and not w.isdigit() and len(w) > 1]
        return x

    def mean_pooling(self, preprocessed_doc):
        embeddings = []
        for word in preprocessed_doc:
            if word in self.model:
                embeddings.append(self.model[word])
        if len(embeddings) > 0:
            final = np.mean(embeddings, axis=0)
        else:
            final = np.zeros(300)
        return final

    def full_process(self, doc: str):
        preprocessed_doc = self.preprocess(doc)
        final_vector = self.mean_pooling(preprocessed_doc)
        return final_vector