from flask import Flask, request, jsonify, render_template
import os
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# ------------------------------
# 1. Initialize Flask
# ------------------------------
app = Flask(__name__)

# ------------------------------
# 2. Load environment variables
# ------------------------------
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# ------------------------------
# 3. Global initialization: Build the VectorStore & QA chain
# ------------------------------
# This runs once at app startup (not on every request).

from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("documents/Atomic habits ( PDFDrive ).pdf")
docs = loader.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
docs = text_splitter.split_documents(docs)

# 3b. Create embeddings & vector store
embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key=openai_api_key
)
vectorstore = Chroma.from_documents(
    docs,
    embedding=embeddings,
    persist_directory="chroma_db"
)

# 3c. Build QA chain
llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    temperature=0.0,
    model_name="gpt-4"  # or "gpt-3.5-turbo"
)

from langchain.prompts import PromptTemplate


# 3d. Custom Prompt
# ------------------------------
prompt_template = """
You are a helpful assistant. You have the following context from the book "Atomic Habits". Provide a detailed response that thoroughly explains the user's question, references important points from the book, and includes examples or anecdotes if relevant. Well-structured, and neatly formatted responses please

Context:
{context}

Question:
{question}

Detailed Answer:
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# ------------------------------
# 3e. Create RetrievalQA chain w/ custom prompt
# ------------------------------
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}  # <-- Provide the custom prompt here
)

#############################################################################
# 4. Flask routes
#############################################################################
@app.route("/", methods=['GET', 'POST'])
def index():
    answer = None
    if request.method == 'POST':
        question = request.form.get('question', '').strip()
        if question:
            result = qa_chain(question)  # same usage, but now uses custom prompt
            answer = result["result"]
            for i, doc in enumerate(result["source_documents"], start=1):
                print(f"--- Source doc {i} ---")
                print(doc.page_content[:300])  # snippet
                print(doc.metadata)

    return render_template('home.html', answer=answer)


#############################################################################
# 5. Run in local dev
#############################################################################
if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
