import gensim.downloader
from nsfw2v_docprocess import document_vector
from fastapi import FastAPI

app = FastAPI()

w2v_vecs = gensim.downloader.load('word2vec-google-news-300')
doc_vec_obj = document_vector(w2v_vecs)

@app.post('/texttovector/')
def create_doc_vector(doc: str):
    embedding = doc_vec_obj.full_process(doc)
    embedding = embedding.tolist()
    return embedding
