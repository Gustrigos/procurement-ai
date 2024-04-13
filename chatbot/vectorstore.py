from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from langchain.document_loaders import PyPDFLoader

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import logging
import time

import requests
from tempfile import NamedTemporaryFile

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class XrefWarningCounterHandler(logging.Handler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.xref_warnings = 0
        self.limit = 20  # Set this to whatever limit you prefer

    def handle(self, record):
        if "Xref table invalid" in record.getMessage():
            self.xref_warnings += 1
            if self.xref_warnings >= self.limit:
                # Raising a custom exception when the limit is reached
                raise TooManyXrefWarningsException("Too many Xref table warnings!")

class TooManyXrefWarningsException(Exception):
    pass

class VectorstoreHandler:
    def __init__(self):
        self.setup_logging()
    
    def setup_logging(self):
        logger = logging.getLogger()
        handler = XrefWarningCounterHandler()
        logger.addHandler(handler)  

    def process_pdf_from_url(self, pdf_url):
        response = requests.get(pdf_url)
        if response.status_code == 200:
            # Create a temporary file to store the PDF
            with NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf:
                tmp_pdf.write(response.content)
                tmp_pdf_path = tmp_pdf.name
            
            vectorstore = self.pdf_reader([tmp_pdf_path])
            

            return vectorstore
        else:
            logging.error(f"Failed to download PDF from {pdf_url}. HTTP status code: {response.status_code}")
            return None

    def get_pdf_text(self, pdf_docs):
        start_time = time.time()
        logging.info("Starting to extract text from PDFs.")
        
        text = ""
        for pdf in pdf_docs:
            try:
                pdf_reader = PdfReader(pdf)
                for page in pdf_reader.pages:
                    text += page.extract_text()

            except TooManyXrefWarningsException:
                logging.error(f"Too many Xref table warnings for {pdf}! Skipping the rest of this file.")
                continue  # This will move on to the next PDF in the loop

        duration = time.time() - start_time
        logging.info(f"Finished extracting text from PDFs in {duration:.2f} seconds.")

        return text

    def get_text_chunks(self, text):
        start_time = time.time()
        logging.info("Starting to split text into chunks.")
        
        text_splitter = RecursiveCharacterTextSplitter(
            separators=[" ", ",", "\n", "\n\n"],
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        chunks = text_splitter.split_text(text)
        
        duration = time.time() - start_time
        logging.info(f"Finished splitting text into chunks in {duration:.2f} seconds.")
        return chunks

    def get_vectorstore(self, text_chunks):
        start_time = time.time()
        logging.info("Starting to create vectorstore from text chunks.")
        
        embeddings = HuggingFaceEmbeddings(model_kwargs={'device': 'cpu'})
        vectorstore = Chroma.from_texts(texts=text_chunks, embedding=embeddings)
        
        duration = time.time() - start_time
        logging.info(f"Finished creating vectorstore in {duration:.2f} seconds.")
        return vectorstore

    def pdf_reader(self, pdf_docs):
        start_time = time.time()
        logging.info("Starting the PDF reader process.")
        
        raw_text = self.get_pdf_text(pdf_docs)
        text_chunks = self.get_text_chunks(raw_text)
        logging.info(f"Number of text chunks: {len(text_chunks)}")
        
        vectorstore = self.get_vectorstore(text_chunks)
        
        duration = time.time() - start_time
        logging.info(f"Finished the PDF reader process in {duration:.2f} seconds.")
        return vectorstore

    def pdf_loader(self, pdf_docs):
        start_time = time.time()
        logging.info("Starting to load and split PDF documents.")
        
        loader = PyPDFLoader(pdf_docs)
        pages = loader.load_and_split()
        vectorstore = Chroma.from_documents(pages, collection_name="documents")
        
        duration = time.time() - start_time
        logging.info(f"Finished loading and splitting PDF documents in {duration:.2f} seconds.")
        return vectorstore
    