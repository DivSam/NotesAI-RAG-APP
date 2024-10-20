from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain.vectorstores.chroma import Chroma
from get_embeddings import get_embeddings


DATA_PATH = 'data'
CHROMA_PATH = 'chroma'


def main():
    documents = load_data()
    chunks = chunk_data(documents)
    print(len(chunks))
    print(chunks[0])
    add_documents_to_chroma(chunks)


def load_data() -> list[Document]:
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    return documents


def chunk_data(documents: list[Document]) -> list[list[Document]]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_documents_to_chroma(chunks: list[list[Document]]):
    db = Chroma.from_documents(
        chunks,
        get_embeddings(),
        persist_directory=CHROMA_PATH
    )
    db.persist()


if __name__ == '__main__':
    main()
