import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import Chroma
import chromadb
import uuid

os.environ["OPENAI_API_KEY"] = "your-key"

class Rag_Model:

    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2",
                                                # model_kwargs={"device": "cuda"},
                                                model_kwargs={"device": "cpu"},
                                                encode_kwargs={'normalize_embeddings': True})
        self.retriever = None
        self.emb = "Embeddings"
        self.qa = None
        self.em = None
        self.source = None
        self.persistent_client = None

    def file_loader(self, path):
        try:
            loader = PyMuPDFLoader(path, extract_images=True)
            return loader.load()
        except Exception as e:
            print(e)
            return False

    def text_splitter(self, texts):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,
            chunk_overlap=1000)
        return text_splitter.split_documents(texts)

    def embed_data(self, texts, domain, index):
        self.persistent_client = chromadb.PersistentClient(self.emb + f"/{domain}/")
        collection = self.persistent_client.create_collection(name=index, metadata={"hnsw:space": "cosine"})
        collection = self.persistent_client.get_collection(index)
        for i in texts:
            collection.upsert(
                documents=str(i.page_content).encode('utf-8', 'replace').decode('utf-8'),
                embeddings=self.embeddings.embed_query(i.page_content),
                metadatas=i.metadata,
                ids=[str(uuid.uuid1())]
            )

        langchain_chroma = Chroma(
            client=self.persistent_client,
            collection_name=index,
            embedding_function=self.embeddings, )
        self.retriever = langchain_chroma.as_retriever(search_kwargs={"k": 5})

    def load_embedings(self, domain, index):
        self.persistent_client = chromadb.PersistentClient(self.emb + f"/{domain}/")
        collection = self.persistent_client.get_collection(index)
        langchain_chroma = Chroma(
            client=self.persistent_client,
            collection_name=index,
            embedding_function=self.embeddings,)
        self.retriever = langchain_chroma.as_retriever(search_kwargs={"k": 5})

    def qas(self):
            prompt_template = """You are a helpful and professional AI assistant {context}
            You are provided with a chat history {chat_history} use that history to expand your knowledge about the current conversation that is going on and also use that to keep the conversation helpful and professional
            QUESTION:```{question}```
            ANSWER:"""
            PROMPT = PromptTemplate(
                template=prompt_template, input_variables=["context","question", "chat_history"]
            )
            chain_type_kwargs = {"prompt": PROMPT}
            self.qa = ConversationalRetrievalChain.from_llm(llm=ChatOpenAI(
                temperature=0.5,
                verbose=True,
            ),
                retriever=self.retriever,
                chain_type="stuff",
                combine_docs_chain_kwargs={"prompt": PROMPT},
                return_source_documents=True,
                verbose=True, )

    def output(self, query, history):
        llm_response = self.qa.invoke({"question": query, "chat_history": history})
        json_response = {
            "question": query,
            "answer": llm_response['answer'],
        }
        return json_response
