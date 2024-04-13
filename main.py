from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from chatbot.chatbot import Chatbot
import json 
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost:3000",  
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

class ChatQuery(BaseModel):
    query: str
    model: str = "gpt-4-32k-0613"
    client_name: str = "Mentum"
    custom_instructions: str = ""
    temperature: float = 0.0
    env: str = "prod"
    data_dict: dict = {}

@app.post("/process-message/")
async def chat(query: ChatQuery):
    chatbot = Chatbot(model=query.model, client_name=query.client_name, 
                      custom_instructions=query.custom_instructions, 
                      temperature=query.temperature, env=query.env)
    
    chatbot.set_memory(chatbot.memory)

    try:
        response = chatbot.process_message(query.query)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-pdf/")
async def process_pdf(query: ChatQuery):
    chatbot = Chatbot(model=query.model, client_name=query.client_name, 
                      custom_instructions=query.custom_instructions, 
                      temperature=query.temperature, env=query.env)
    
    chatbot.set_memory(chatbot.memory)
    
    try:
        chatbot.load_data(query.data_dict)
        chatbot.set_memory(chatbot.memory)
        
        response_str = chatbot.process_message(query.query, agent_type="conversational_qa_agent")
        
        # Parse the string response into a Python dictionary
        response_data = json.loads(response_str)
        
        # Return the Python dictionary directly
        return response_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
        
    