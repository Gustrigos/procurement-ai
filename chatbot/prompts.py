from langchain.schema import HumanMessage
from langchain.prompts import BaseChatPromptTemplate
from typing import List
from langchain.agents import Tool
from langchain import PromptTemplate

from .templates import Template

class Prompt:
    def __init__(self, client_name, tools, custom_instructions):
        self.template = Template(client_name, tools, custom_instructions)
        self.client_name = client_name
        self.tools = tools
        self.custom_instructions = custom_instructions

    def initialize_prompt(self,tool_prompt=False):
        identity_template = self.template.identity_template()
        instructions_template = self.template.instructions_template()
        plot_instructions_template = self.template.plot_instructions_template()
        custom_instructions_template = self.template.custom_instructions_template()
        format_template = self.template.format_template()

        all_instructions_template = f"{identity_template} {instructions_template} {plot_instructions_template}"
        tool_prompt_template = f"{identity_template} {plot_instructions_template}"

        if self.custom_instructions:
            tool_prompt_template += custom_instructions_template
            all_instructions_template += custom_instructions_template

        if tool_prompt:
            final_template = f"{tool_prompt_template} {format_template}"
        else:
            final_template = f"{all_instructions_template} {format_template}"

        prompt = CustomPromptTemplate(
            template=final_template,
            tools=self.tools,
            input_variables=["input", "intermediate_steps", "chat_history"],
            client_name=self.client_name
        )

        return prompt
    
    def initialize_conversational_qa_prompt(self):
        conversational_qa_identity_template = self.template.conversational_qa_identity_template()
        conservational_qa_instructions_template = self.template.conservational_qa_instructions_template()
        plot_instructions_template = self.template.plot_instructions_template()
        condense_question_template = self.template.condense_question_template()

        condense_question_prompt_template = PromptTemplate(
            template=condense_question_template,
            input_variables=["question", "chat_history"],
        )
        
        qa_prompt_template = f"{conversational_qa_identity_template} {conservational_qa_instructions_template} {plot_instructions_template}"
        qa_prompt = PromptTemplate(template=qa_prompt_template, input_variables=["context", "question"])

        return condense_question_prompt_template, qa_prompt

class CustomPromptTemplate(BaseChatPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]
    # Client's name
    client_name: str

    def format_messages(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])

        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        # client name
        kwargs["client_name"] = self.client_name
        formatted = self.template.format(**kwargs)
        return [HumanMessage(content=formatted)]
    