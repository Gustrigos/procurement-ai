from langchain import LLMChain

class ChainManager:
    def __init__(self, llm, prompt, tools=None, memory=None):
        self.llm = llm
        self.prompt = prompt
        self.tools = tools
        self.memory = memory

    def initialize_chain(self):
        llm_chain = LLMChain(llm=self.llm, prompt=self.prompt, memory=self.memory, verbose=True)
        
        return llm_chain

    # def initialize_tool_chain(self):
    #     llm_chain = ToolChain(prompt=self.prompt, tools=self.tools, verbose=True)
        
    #     return llm_chain
    
