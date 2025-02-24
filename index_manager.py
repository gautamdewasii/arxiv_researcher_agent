from tools import fetch_arxiv_papers
from llama_index.core import Document, Settings, VectorStoreIndex, StorageContext, load_index_from_storage

class IndexManager:
    def __init__(self, embed_model):
        self.embed_model = embed_model
        self.papers = []

    def fetch_papers(self, topic, papers_count=10):
        self.papers = fetch_arxiv_papers(topic, papers_count)

    def create_documents_from_papers(self):
        for paper in self.papers:
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
            self.documents.append(Document(text=content))

    def create_index(self):
        self.documents = []
        self.create_documents_from_papers()
        Settings.chunk_size = 1024
        Settings.chunk_overlap = 50

        index = VectorStoreIndex.from_documents(self.documents, embed_model=self.embed_model)
    
    def retrieve_index(self):
        storage_context = StorageContext.from_defaults(persist_dir="index/")
        return load_index_from_storage(storage_context, embed_model=self.embed_model)

    def list_papers(self):
        print([paper["title"] for paper in self.papers])