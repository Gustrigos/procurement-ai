from langchain.agents import Tool

class ToolManager:
    def __init__(self, llm, data_dict):
        self.llm = llm
        self.data_dict = data_dict
        self.tools = []  # Initialize as an empty list

    def initialize_tool(self, tool_type, **kwargs):
        if tool_type == 'general_search':
            return self._initialize_search_tool()
        elif tool_type == 'conversational_qa_agent':
            return self.initialize_conversational_qa_tool(kwargs.get('conversational_qa_agent'), kwargs.get('vectorstore_name'), kwargs.get('tool_description'))
        # Add other conditions here for different tool types

    def add_tool(self, name, func, description):
        # Remove the previous tool with the same name if it exists
        self.tools = [tool for tool in self.tools if tool.name != name]

        self.tools.append(
            Tool(
                name=name,
                func=func,
                description=description
            )
        )

    def initialize_conversational_qa_tool(self, conversational_qa_agent, vectorstore_name, tool_description):
        # import chain and agent 
        tool_name = f"{vectorstore_name.capitalize()} Question Retrieval Agent"
        self.add_tool(
            name=tool_name,
            func=conversational_qa_agent.run,
            description=f"{tool_description}"
        )
        