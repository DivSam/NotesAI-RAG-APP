from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document


DATA_PATH = 'data'


def main():
    documents = load_data()
    chunks = chunk_data(documents)
    print(len(chunks))
    print(chunks[0])


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


if __name__ == '__main__':
    main()
