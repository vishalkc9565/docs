import time
from fastapi import APIRouter, HTTPException
import os
import random
from . import db
from ...copilot.db import (
    create_new_session,
    fetch_session_details,
)
from ...models.chat import (
    ChatRequest,
    ChatResponse,
    CreateSessionRequest,
    CreateSessionResponse,
)
from retell import Retell
from ...models import HealthSuccess
from ...copilot.assistant import (
    run_assistant,
    create_thread,
    create_user_message,
)
from ...copilot.exceptions import (
    ToolFailureException,
    SessionNotFoundException,
    AssistantRunFailedException,
)
from ...utils.logger import get_logger

import request
from fastapi.responses import JSONResponse as jsonify
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


logger = get_logger(__name__)
file_chat_api_router = APIRouter()


@file_chat_api_router.get(
    "/health",
    tags=["Health"],
    status_code=200,
    summary="Health Check",
    description="A simple health check endpoint to ensure the service is running.",
)
async def health_check():
    return HealthSuccess(message='File-Health: OK')



@file_chat_api_router.post('/process_file' )
def process_pdf():
    if 'files' not in request.files:
        return jsonify({'error': 'No PDF files uploaded'})
    
    files = request.files.getlist('files')
    if len(files) == 0:
        return jsonify({'error': 'No PDF files uploaded'})
    
    text = ""
    for pdf_file in files:
        # Check if file is PDF
        if pdf_file.filename.endswith('.pdf'):
            # Read PDF content
            pdf_reader = PdfReader(pdf_file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        else:
            return jsonify({'error': 'Uploaded file is not a PDF'})
    
    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    
    # Get embeddings for each chunk
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    
    return jsonify({'success': True})
 
@file_chat_api_router.route('/chatbot', methods=['POST'])
def chatbot():
    user_input = request.json.get('user_input')
    
    if not user_input:
        return jsonify({'error': 'No user input provided'})
    
    # Define the prompt template for the question answering model
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    # Load the conversational chain
    model = ChatGoogleGenerativeAI(model="gemini-pro", client=genai, temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    
    # Run the chatbot with user input
    response = chain({"context": "", "question": user_input}, return_only_outputs=True)

    return jsonify({'response': response['output_text']})
