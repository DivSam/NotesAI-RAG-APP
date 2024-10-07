from generate_database import CHROMA_PATH
from langchain.vectorstores.chroma import Chroma
from get_embeddings import get_embeddings

PROMPT = """
Answer the question based solely on this context:
{context}

====================

Answer the question based on the above context: {question}
"""


def query_vector_database(query: str):
    db = Chroma(persist_directory=CHROMA_PATH,
                embedding_function=get_embeddings())

    results = db._similarity_search_with_relevance_scores(query, k=3)

    return results
