import os
from dotenv import load_dotenv
import traceback

from langchain.memory import ConversationBufferWindowMemory

# repo imports
from .models import ModelManager
from .tools import ToolManager
from .output_parsers import CustomOutputParser
from .agents import AgentManager

import json
import logging

# Determine the environment
ENVIRONMENT = os.getenv("ENVIRONMENT", "prod")  # default to local if not set

# OpenAI setup
env_file_path = ".env.local"
load_dotenv(env_file_path)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

from chatbot.vectorstore import VectorstoreHandler

class Chatbot:
    def __init__(self, model, client_name, custom_instructions, temperature, env):
        self.client_name = client_name
        self.custom_instructions = custom_instructions
        self.temperature = temperature
        self.model = model
        self.data_dict = {} 
        self.tags = [model, client_name, custom_instructions, env]
        self.metadata = {}

        # Instantiate Model
        self.model_manager = ModelManager(model_type='ChatOpenAI', streaming=True, callbacks=None, model_name=self.model, temperature=temperature)
        self.llm = self.model_manager.initialize_model_type()

        # Instantiate Tools
        self.tool_manager = ToolManager(self.llm, self.data_dict)
        # Add new tools here...
        self.tools = self.tool_manager.tools

        self.output_parser = CustomOutputParser()

        # Default memories
        self.memory = self.get_memory()
        self.conversational_qa_memory = self.get_memory(input_key="question", k=5)

        self.agent_manager = AgentManager(self.llm, self.tools, self.output_parser, client_name, custom_instructions, tags=self.tags, metadata=self.metadata)

    def get_memory(self, input_key="input", k=5):
        return ConversationBufferWindowMemory(
            memory_key="chat_history",
            input_key=input_key,
            return_messages=True,
            k=k
        )

    def set_memory(self, memory_type="default", **kwargs):
        # setting memory from state for client to remember conversations.
        self.agent_manager.set_memory(memory_type, **kwargs)
    
    def load_data(self, data_dict):
        self.data_dict = data_dict

        pdf_description = None
        pdf_metadata = []

        # Pass the updated data_dict to the tool_manager
        self.tool_manager.data_dict = self.data_dict

        # Extract PDFs directly or use an empty list as default
        pdfs = getattr(data_dict, 'pdfs', [])
        pdf_urls = [pdf.pdfUrl for pdf in pdfs if hasattr(pdf, 'pdfUrl')]

        if pdf_urls:
            vectorstore_handler = VectorstoreHandler()

            # Process multiple PDF URLs
            vectorstore = vectorstore_handler.process_pdfs_from_urls(pdf_urls)
            
            # Collect descriptions and metadata if available
            pdf_descriptions = [pdf.description for pdf in pdfs if hasattr(pdf, 'description')]
            pdf_description = " ".join(pdf_descriptions)  # Concatenate descriptions
            
            # Collect metadata if available, handle it properly assuming it could be a list of objects or dicts
            for pdf in pdfs:
                if hasattr(pdf, 'metadata') and pdf.metadata is not None:
                    pdf_metadata.append(pdf.metadata)

            if vectorstore is not None:
                conversational_qa_agent, retriever = self.agent_manager.initialize_conversational_qa_agent(vectorstore, memory=self.conversational_qa_memory)
                
                # Initialize Conversational QA Tool
                pdf_tool_description = "Useful for answering questions related to PDF files that can be contracts, invoices, and other financial documents."
                
                for index, metadata in enumerate(pdf_metadata, start=1):
                    pdf_tool_description += f" The file name of Document {index} is {getattr(metadata, 'name', 'unknown')}, and its file type is {getattr(metadata, 'type', 'unknown')}."

                if pdf_description:
                    pdf_tool_description += f" The documents are about: {pdf_description}."

                self.tool_manager.initialize_tool('conversational_qa_agent', conversational_qa_agent=conversational_qa_agent, vectorstore_name='PDF', tool_description=pdf_tool_description)

        # Update the tools list
        self.tools = self.tool_manager.tools
        tool_names = [tool.name for tool in self.tools]

        self.metadata = {
            "pdf_description": pdf_description,
            "pdf_metadata": pdf_metadata,
            "tool_names": tool_names
        }     

        # Re-initialize the agent and executor with the updated tools
        self.agent_manager.reinitialize_agent_and_executor(tools=self.tools, metadata=self.metadata)

    def set_model(self, model):
        self.model = model

    def set_temperature(self, temperature):
        self.temperature = temperature

    def set_client_name(self, client_name):
        self.client_name = client_name 

    def set_custom_instructions(self, custom_instructions):
        self.custom_instructions = custom_instructions 

    def process_message(self, user_prompt, callbacks=None, agent_type=None):
        
        try:
            result = self.agent_manager.process_message(user_prompt, callbacks, agent_type)
            return result
        except Exception as e:
            logging.error(f"Failed to process message: {str(e)}")
            logging.error(f"Data received: '{user_prompt}'")  # Log the input data that caused the error
            logging.error(traceback.format_exc())  # Include the traceback to see where the error occurred
            return json.dumps({"error": "Internal server error", "detail": str(e)})
        
