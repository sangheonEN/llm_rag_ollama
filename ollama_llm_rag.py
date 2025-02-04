from ollama import chat
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

import argparse
import os
import time


parser = argparse.ArgumentParser()
parser.add_argument("--text", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag_text.txt"))
args = parser.parse_args()


def text_loader(text):
    loader = TextLoader(text, encoding="utf-8")
    data = loader.load()

    return data


def text_spliter(text):
    
    text_spliter = CharacterTextSplitter(
        separator='',
        chunk_size=500,
        chunk_overlap = 100,
        length_function = len,
    )
    
    texts = text_spliter.split_text(text[0].page_content)
    
    return texts


def vector_embedding():
    
    embeddings_model = HuggingFaceEmbeddings(
    model_name='jhgan/ko-sroberta-nli',
    model_kwargs={'device':'cuda'},
    encode_kwargs={'normalize_embeddings':True},
    )

    return embeddings_model


def process_time_cal(task_name, start_time):

    end_time = time.process_time()
    print(f"task_name: {task_name}")
    print(f"time elapsed : {int(round((end_time - start_time) * 1000))}ms")

if __name__ == "__main__":
    
    start_time = time.process_time()
    
    data = text_loader(args.text)
    print(f"data: {data[0].page_content}")
    process_time_cal("text_loader", start_time)
    
    
    start_time = time.process_time()
    split_data = text_spliter(data)
    print(f"\nsplit_data: {split_data}\n")
    print(f"\nsplit_data len : {len(split_data)}\n")
    process_time_cal("text_spliter", start_time)
    
    start_time = time.process_time()
    embedding_model = vector_embedding()
    process_time_cal("vector_embedding", start_time)
    
    start_time = time.process_time()
    vector = FAISS.from_texts(split_data, embedding_model)
    print(f"vector.index.ntotal: \n {vector.index.ntotal}\n")
    print(f"vector = FAISS.from_texts(split_data, embedding_model): \n {vector}\n")
    process_time_cal("FAISS.from_texts", start_time)
    
    start_time = time.process_time()
    retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    print(f"vector.as_retriever: \n {retriever}\n")

    process_time_cal("vector.as_retriever", start_time)

    start_time = time.process_time()
    llm = Ollama(model="deepseek_r1_14b")
    process_time_cal("Ollama(model=deepseek_r1_14b)", start_time)

    prompt = """
                Use the following context to answer the question:
                Context: {context}
                Question: {question}
                Answer:
            """
            
    start_time = time.process_time()
    QA_PROMPT = PromptTemplate.from_template(prompt)
    print(f"\nQA_PROMPT: {QA_PROMPT}\n")
    process_time_cal("PromptTemplate.from_template", start_time)
    
    start_time = time.process_time()
    llm_chain = LLMChain(llm=llm, prompt=QA_PROMPT)
    process_time_cal("LLMChain(llm=llm, prompt=QA_PROMPT)", start_time)
    
    start_time = time.process_time()
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=llm_chain, document_variable_name="context"
    )
    process_time_cal("StuffDocumentsChain", start_time)
    
    start_time = time.process_time()
    qa = RetrievalQA(combine_documents_chain=combine_documents_chain, retriever=retriever)
    process_time_cal("RetrievalQA", start_time)
    
    start_time = time.process_time()
    response = qa.run({"query": "환자분 안녕하세요"})
    process_time_cal("qa.run", start_time)
        
    print("\nResponse:")
    print(response)