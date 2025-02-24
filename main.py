from tools import fetch_arxiv_papers
from llama_index.core import Document
from llama_index.core import Settings, VectorStoreIndex
from constants import embed_model

papers = fetch_arxiv_papers("Large language models", 10)
def create_documents_from_papers(papers):
    documents = []
    for paper in papers:
        content = (
            f"title: {paper['title']}\n"
            f"authors: {paper['authors']}\n"
            f"summary: {paper['summary']}\n"
            f"published: {paper['published']}\n"
            f"journal_ref: {paper['journal_ref']}\n"
            f"doi: {paper['doi']}\n"
            f"primary_category: {paper['primary_category']}\n"
            f"categories: {','.join(paper['categories'])}\n"
            f"pdf_url: {paper['pdf_url']}\n"
            f"arxiv_url: {paper['arxiv_url']}\n"
        )
        documents.append(Document(text=content))
    return documents

documents = create_documents_from_papers(papers)

Settings.chunk_size = 1024
Settings.chunk_overlap = 50

index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
index.storage_context.persist("index/")