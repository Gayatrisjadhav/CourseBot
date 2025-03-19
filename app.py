from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain import hub
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

model_name = "BAAI/bge-large-en"
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)

groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model= "llama-3.3-70b-versatile", api_key=groq_api_key)

loader = WebBaseLoader("https://brainlox.com/courses/category/technical")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

vectorstore = FAISS.from_documents(documents=splits, embedding=hf)
retriever = vectorstore.as_retriever()

prompt = hub.pull("rlm/rag-prompt", api_url="https://api.hub.langchain.com")
print(prompt)
qa_chain = RetrievalQA.from_chain_type(
    llm, retriever=retriever, chain_type_kwargs={"prompt": prompt}
)
@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get("question")
    if not question:
        return jsonify({"error": "Please provide a 'question' in the request body."}), 400
    result = qa_chain({"query": question})
    return jsonify({"answer": result["result"]})

if __name__ == '__main__':
    app.run(debug=True)