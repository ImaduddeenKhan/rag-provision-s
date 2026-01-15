import os
import logging
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler("rag_debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)
UPLOAD_FOLDER = 'temp_pdfs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

logger.info("Initializing models")
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash") 
logger.info("Models initialized successfully")

vectorstore = None

@app.route('/')
def index():
    logger.info("Home page accessed")
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    global vectorstore
    logger.info("Upload request received")
    files = request.files.getlist('pdfs')
    logger.info(f"Found {len(files)} files to process")
    
    all_docs = []
    for file in files:
        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)
        logger.info(f"Saved file: {file.filename}")
        loader = PyPDFLoader(path)
        docs = loader.load()
        all_docs.extend(docs)
        logger.info(f"Loaded {len(docs)} pages from {file.filename}")
    
    logger.info(f"Starting chunking: {len(all_docs)} total pages")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    final_chunks = text_splitter.split_documents(all_docs)
    logger.info(f"Created {len(final_chunks)} chunks")
    
    logger.info("Creating vectorstore")
    vectorstore = Chroma.from_documents(
        documents=final_chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    logger.info("Vectorstore created successfully")
    
    return jsonify({"status": "Success", "message": f"Processed {len(files)} files."})

@app.route('/chat', methods=['POST'])
def chat():
    user_query = request.json.get('query')
    logger.info(f"Chat request received: '{user_query[:100]}...'")
    
    if not vectorstore:
        logger.error("No vectorstore available - upload documents first")
        return jsonify({"error": "Please upload PDFs first!"})
    
    try:
        
        logger.info("Retrieving top 3 documents")
        docs = vectorstore.similarity_search(user_query, k=3)
        logger.info(f"Retrieved {len(docs)} documents")
        
       
        logger.info("Cleaning document context")
        cleaned_context = ""
        for i, doc in enumerate(docs):
            clean_text = " ".join(doc.page_content.split())
            cleaned_context += f"Chunk {i+1}:\n{clean_text}\n\n"
        logger.info(f"Cleaned context length: {len(cleaned_context)} characters")
        
        logger.info("Creating prompt template")
        system_prompt = (
        "You are an expert legal assistant specializing in USA State Provision files. "
        "Your goal is to provide a direct and concise answer to the user's question based ONLY on the context below.\n\n"
        "Rules for your response:\n"
        "1. Analyze the context to find the specific rule that matches the user's query.\n"
        "2. If the context specifies different rules for different conditions (like width or length), mention those clearly but briefly.\n"
        "3. Do not use phrases like 'Based on the provided context.'\n"
        "4. Format the output line by line for readability.\n"
        "5. If the answer is missing, say 'I do not have this information.'\n\n"
        "CONTEXT:\n{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        logger.info("Formatting final prompt")
        formatted_prompt = prompt.format(context=cleaned_context, input=user_query)
        logger.info(f"Final prompt length: {len(formatted_prompt)} characters")
        
        logger.info("Calling LLM for response generation")
        response = llm.invoke(formatted_prompt)
        logger.info("LLM response received successfully")
        logger.info(f"Response preview: {response.content[:200]}...")
        
        return jsonify({"answer": response.content})
    
    except Exception as e:
        logger.error(f"Error in chat processing: {str(e)}")
        logger.error(f"Full traceback:", exc_info=True)
        return jsonify({"answer": f"Error: {str(e)}"})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port)
