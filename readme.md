[ START APPLICATION ]
                 |
        ( Initialize Models )
    - Gemini-2.5-Flash (LLM)
    - Text-Embedding-004 (Embeddings)
                 |
                 V
+---------------------------------------+       +---------------------------------------+
|        ROUTE: /upload (POST)          |       |          ROUTE: /chat (POST)          |
+---------------------------------------+       +---------------------------------------+
| 1. Receive PDF files                  |       | 1. Receive User Query                 |
| 2. Save to 'temp_pdfs'                |       | 2. Check: retrievers initialized?     |
| 3. Load PDFs (PyPDFLoader)            |       |           |                           |
| 4. Split Text (600/150 overlap)       |       |           V                           |
| 5. Generate Vector Embeddings         |       | 3. [ HYBRID SEARCH EXECUTION ]        |
| 6. Save to ChromaDB (Vector Store)    |       |    /---------------------------\      |
| 7. Create BM25 Index (Local Keyword)  |       |    |   Vector   |     BM25      |      |
|                                       |       |    | Search (k=2)| Search (k=2) |      |
|          [ INDEX READY ]              |       |    \-------------+--------------/      |
+---------------------------------------+       |           |                           |
                 |                              |    ( Ensemble & Re-ranking )          |
                 |                              |           |                           |
                 |                              |           V                           |
                 |                              | 4. Merge Unique Top Documents         |
                 |                              | 5. Clean & Format Context Text        |
                 |                              | 6. Inject into System Prompt          |
                 |                              | 7. Call Gemini-2.5-Flash (One Call)   |
                 |                              |           |                           |
                 |                              |           V                           |
                 |                              | 8. Return JSON Answer to User         |
                 +----------------------------->+---------------------------------------+
                                        |
                                [ LOGGING & DEBUG ]
                       (rag_debug.log / Stream Handler)

code updated 16 jan 2026.. "applied hybrid search(vector search + BM25 words matching)"





1. The Project Process (Simplified)
The process is divided into two main phases: Ingestion (preparing the data) and Inference (answering the question).

**Phase A: Data Ingestion (Saving PDFs to DB)**
- Upload & Load: The user selects multiple PDFs in the UI. The upload_files function saves these to a folder on your laptop. We use PyPDFLoader.load() to extract the raw text from each page.
- Chunking: Large documents are split into smaller pieces (chunks) so they fit into the AI's memory. We use RecursiveCharacterTextSplitter.split_documents() for this.
- Vectorization & Storage: Each chunk is turned into a list of numbers (embeddings) and saved into a local folder. We use Chroma.from_documents() to create this searchable "math map" of your documents.

**Phase B: Retrieval & Chat (Answering Questions)**
- Search: When the user asks a question, we use vectorstore.similarity_search() to find the top 5 most relevant chunks from the database.
- Manual Cleaning: Instead of sending everything, we loop through the 5 chunks, extract only the text (doc.page_content), remove extra whitespaces, and add them together line by line.
- Generation: This clean text is placed inside a prompt and sent to the LLM using llm.invoke(). The model reads only the relevant facts and gives you the final answer.




**2. How to Improve Latency (Speed)**
- Latency is the time it takes for the user to get an answer. You can improve it in several ways:
- Use a Faster Model: Ensure you are using Gemini 1.5 Flash (or 2.0 Flash) instead of "Pro" models. The "Flash" versions are specifically designed for low-latency, real-time applications.
- Enable Streaming: Instead of waiting for the full paragraph to be ready, you can show the user the response word-by-word as it is generated. In LangChain, set streaming=True in your LLM configuration.
- Reduce Top-K: Instead of retrieving 5 chunks, try retrieving 3 (k=3). Less text means faster processing by the LLM and lower token usage.
- Optimized Indexing: For local storage like Chroma, using a fast indexing method like HNSW (which it uses by default) is good, but you can also try InMemoryVectorStore if your data is small enough to fit in RAM for even faster lookups.
- Persistent Connection: Avoid re-initializing the embedding model or vector store on every request. Keep the vectorstore object global so it stays in memory after the first upload.

