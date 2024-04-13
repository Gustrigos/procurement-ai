from langchain.chat_models import ChatOpenAI

class ModelManager:
    def __init__(self, model_type='ChatOpenAI', streaming=True, callbacks=None, model_name=None, temperature=None):
        self.model_type = model_type
        self.streaming = streaming
        self.callbacks = callbacks
        self.model_name = model_name
        self.temperature = temperature
        tags=None

    def initialize_model_type(self):
        if self.model_type == 'ChatOpenAI':
            return self._initialize_chat_openai()
        # You can add other conditions here for different models

    def _initialize_chat_openai(self):
        return ChatOpenAI(streaming=self.streaming, callbacks=self.callbacks, model_name=self.model_name, temperature=self.temperature)
    