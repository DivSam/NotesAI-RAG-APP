from generate_database import CHROMA_PATH
from langchain.vectorstores.chroma import Chroma
from get_embeddings import get_embeddings
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

PROMPT = """
Answer the question based solely on this context:
{context}

====================

Answer the question based on the above context: {question}
"""

GENERATE_QUESTION_PROMPT = """
Based on the following context and your knowledge, generate 15 multiple choice questions and answers with 4 choices each and the correct answer always being the first choice.
----------------------
{context}
"""


def query_vector_database(query: str) -> list[dict]:
    db = Chroma(persist_directory=CHROMA_PATH,
                embedding_function=get_embeddings())

    results = db._similarity_search_with_relevance_scores(query, k=3)

    return results


def query_model(results: list[dict]) -> str:
    context = "\n\n --- \n\n".join([doc.page_content for doc, _ in results])

    prompt_template = ChatPromptTemplate.from_template(
        GENERATE_QUESTION_PROMPT)

    prompt = prompt_template.format(context=context)
    print(prompt)

    model = ChatOpenAI()
    response_text = model.predict(prompt)

    return response_text
