import os
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
# We keep these for the generation chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

app = Flask(__name__)
UPLOAD_FOLDER = 'temp_pdfs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 1. Models - Using Gemini 2.0 (Note: Ensure version compatibility)
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash") # Use 1.5-flash for reliability or 2.0-flash-exp

vectorstore = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    global vectorstore
    files = request.files.getlist('pdfs')
    all_docs = []
    
    for file in files:
        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)
        loader = PyPDFLoader(path)
        all_docs.extend(loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    final_chunks = text_splitter.split_documents(all_docs)
    
    vectorstore = Chroma.from_documents(
        documents=final_chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    return jsonify({"status": "Success", "message": f"Processed {len(files)} files."})

@app.route('/chat', methods=['POST'])
def chat():
    user_query = request.json.get('query')
    if not vectorstore:
        return jsonify({"error": "Please upload PDFs first!"})
    
    # --- LOGIC UPGRADE: MANUAL CLEANING ---
    # 1. Retrieve the top 5 documents manually
    docs = vectorstore.similarity_search(user_query, k=5) 
    
    # 2. Extract only the text and clean it (remove extra spaces/junk)
    # We join them line by line as you requested
    cleaned_context = ""
    for i, doc in enumerate(docs):
        # Remove extra whitespace and leading/trailing junk
        clean_text = " ".join(doc.page_content.split()) 
        cleaned_context += f"Chunk {i+1}:\n{clean_text}\n\n"

    # 3. Create the prompt with ONLY the cleaned text
    system_prompt = (
        "You are a Provision file AI assistant. Answer using ONLY the provided context.\n"
        "These files are provision files for different USA states. Answer accurately.\n"
        "If the answer is not in the context, say 'I do not have this information.'\n\n"
        "CONTEXT:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 4. Direct invocation of the LLM with the formatted prompt
    # This bypasses the default chain's habit of sending metadata
    formatted_prompt = prompt.format(context=cleaned_context, input=user_query)
    response = llm.invoke(formatted_prompt)
    
    return jsonify({"answer": response.content})

if __name__ == '__main__':
    # Railway provides a 'PORT' environment variable automatically
    port = int(os.environ.get("PORT", 5000))
    # host='0.0.0.0' is required for cloud deployment
    app.run(host='0.0.0.0', port=port)