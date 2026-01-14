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

