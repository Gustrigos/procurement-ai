from langchain.agents import LLMSingleActionAgent, AgentExecutor
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain

from .templates import Template

from .chains import ChainManager
from .prompts import Prompt
# Add any other necessary imports here

class AgentManager:
    def __init__(self, llm, tools, output_parser, client_name, custom_instructions, tags=None, metadata=None):
        self.llm = llm
        self.tools = tools
        self.output_parser = output_parser
        self.client_name = client_name
        self.custom_instructions = custom_instructions
        self.tags = tags
        self.metadata = metadata
        self.template = Template(client_name, tools, custom_instructions)
        self.initialize_agent_and_executor()

    def initialize_agent_and_executor(self):
        prompt = Prompt(self.client_name, self.tools, self.custom_instructions)
        self.prompt = prompt.initialize_prompt()

        chain_manager = ChainManager(llm=self.llm, prompt=self.prompt)
        self.llm_chain = chain_manager.initialize_chain()
    
        tool_names = [tool.name for tool in self.tools]
        
        self.agent = LLMSingleActionAgent(
            llm_chain=self.llm_chain, 
            output_parser=self.output_parser,
            stop=["\nObservation:"], 
            allowed_tools=tool_names
        )

        self.agent_executor = AgentExecutor.from_agent_and_tools(agent=self.agent, tools=self.tools, verbose=True, handle_parsing_errors=True)

    def initialize_conversational_qa_agent(self, vectorstore, memory=None):
        
        condense_question_prompt_template, qa_prompt = Prompt(self.client_name, self.tools, self.custom_instructions).initialize_conversational_qa_prompt()
        
        chain_manager = ChainManager(llm=self.llm, prompt=condense_question_prompt_template, memory=memory)
        question_generator = chain_manager.initialize_chain()

        doc_chain = load_qa_chain(self.llm, chain_type="stuff", prompt=qa_prompt, verbose=True)

        ## Debugging with relevant documents
        retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
        # print("relevant docs",vectorstore.as_retriever().get_relevant_documents("what are the latest transactions for fidelity and Larrain Vial?"))

        conversational_qa_agent = ConversationalRetrievalChain(
                retriever=retriever,
                question_generator=question_generator,
                combine_docs_chain=doc_chain,
                memory=memory,
                verbose=True
            )
        
        self.conversational_qa_agent = conversational_qa_agent
        
        return conversational_qa_agent, retriever

    def reinitialize_agent_and_executor(self, tools, metadata):
        self.tools = tools
        self.metadata = metadata
        self.initialize_agent_and_executor()

    def process_message(self, user_prompt, callbacks=None, agent_type=None):
        
        selected_agent = self.agent_executor

        if agent_type == 'conversational_qa':
            selected_agent = self.conversational_qa_agent

        if callbacks:
            return selected_agent.run(user_prompt, callbacks=callbacks, tags=self.tags, metadata=self.metadata)
        else:
            return selected_agent.run(user_prompt, tags=self.tags, metadata=self.metadata)

    def set_memory(self, memory):
        # Setting memory from state for client to remember conversations.
        self.agent_executor.memory = memory
        
