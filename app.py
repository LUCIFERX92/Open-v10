import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceEndpoint
from pdfminer.high_level import extract_text
import docx2txt
import os
import re
from typing import List
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceEndpoint
from pdfminer.high_level import extract_text
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
import os
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

HUGGINGFACEHUB_API_TOKEN = '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'
os.environ['HUGGINGFACEHUB_API_TOKEN'] = HUGGINGFACEHUB_API_TOKEN

def extract_text_from_pdf(pdf_path):
    return extract_text(pdf_path)

def extract_text_from_doc(doc_path):
    return docx2txt.process(doc_path)

def preprocess_text(text):
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_files(file_paths: List[str]):
    all_text = ""
    for file_path in file_paths:
        print(file_path)
        if file_path.endswith(".pdf"):
            extracted_text = extract_text_from_pdf(file_path)
        elif file_path.endswith(".doc") or file_path.endswith(".docx"):
            extracted_text = extract_text_from_doc(file_path)
        else:
            print(f"Unsupported file type: {file_path}")
            continue
        preprocessed_text = preprocess_text(extracted_text)
        all_text += preprocessed_text + " "
    return all_text

def compute_cosine_similarity_scores(query, retrieved_docs):
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    query_embedding = model.encode(query, convert_to_tensor=True)
    doc_embeddings = model.encode(retrieved_docs, convert_to_tensor=True)
    cosine_scores = np.dot(doc_embeddings, query_embedding.T)
    readable_scores = [{"doc": doc, "score": float(score)} for doc, score in zip(retrieved_docs, cosine_scores.flatten())]
    return readable_scores

def answer_query_with_similarity(query, file_paths):
    try:
        all_text = process_files(file_paths)

        embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_text(all_text)

        vector_store = Chroma.from_texts(texts, embeddings, collection_metadata={"hnsw:space": "cosine"}, persist_directory="stores/insurance_cosine")
        load_vector_store = Chroma(persist_directory="stores/insurance_cosine", embedding_function=embeddings)
        print("Vector DB Successfully Created!")

        db3 = Chroma(persist_directory=f"stores/insurance_cosine", embedding_function=embeddings)
        docs = db3.similarity_search(query)
        print(f"\n\nDocuments retrieved: {len(docs)}")

        if not docs:
            print("No documents match the query.")
            return None, None

        docs_content = [doc.page_content for doc in docs]
        for i, content in enumerate(docs_content, start=1):
            print(f"\nDocument {i}: {content}...")

        cosine_similarity_scores = compute_cosine_similarity_scores(query, docs_content)
        for score in cosine_similarity_scores:
            print(f"\nDocument Score: {score['score']}")

        all_docs_content = " ".join(docs_content)

        template = """
                ### [INST] Instruction:Analyze the provided PDF and DOC documents focusing specifically on extracting factual content, mathematical data, and crucial information relevant to device specifications, including discription. Utilize the RAG model's retrieval capabilities to ensure accuracy and minimize the risk of hallucinations in the generated content. Present the findings in a structured and clear format, incorporating:

                    Device Specifications: List all relevant device specifications, including batch numbers, ensuring accuracy and attention to detail.
                    Mathematical Calculations: Perform and report any necessary mathematical calculations found within the documents, providing step-by-step explanations to ensure clarity.
                    Numerical Data Analysis: Extract and analyze numerical data from tables included in the documents, summarizing key findings and implications.
                    Factual Information: Highlight crucial factual information extracted from the text, ensuring it is presented in a straightforward and understandable manner.
                    Ensure the response is well-organized, using bullet points or numbered lists where applicable, to enhance readability and presentation. Avoid any form of hallucination by cross-referencing facts with the document content directly.

                ### Docs : {docs}
                ### Question : {question}
                """
        prompt = PromptTemplate.from_template(template.format(docs=all_docs_content, question=query))

        repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        llm = HuggingFaceEndpoint(repo_id=repo_id, temperature=0.1, token=HUGGINGFACEHUB_API_TOKEN,
                                  top_p=0.15,
                                  max_new_tokens=512,
                                  repetition_penalty=1.1
                                  )
        llm_chain = LLMChain(prompt=prompt, llm=llm)

        answer = llm_chain.run(question=query)
        cleaned_answer = answer.split("Answer:")[-1].strip()
        print(f"\n\nAnswer: {cleaned_answer}")

        return cleaned_answer,
    except Exception as e:
        print("An error occurred to get the answer: ", str(e))
        return None, None

def main():
    st.title("Document Query App")

    # Get user inputs
    file_paths = st.text_input("Enter the file paths (comma-separated):")
    file_paths = [path.strip() for path in file_paths.split(",")]

    query = st.text_input("Enter your query:")

    if st.button("Get Answer"):
        if file_paths and query:
            response = answer_query_with_similarity(query, file_paths)
            if response:
                st.write("Answer:", response[0])
            else:
                st.write("No answer found.")
        else:
            st.write("Please provide file paths and a query.")
