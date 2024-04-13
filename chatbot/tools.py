from langchain.agents import Tool
from langchain import SerpAPIWrapper
from langchain.agents.agent_toolkits import create_pandas_dataframe_agent

class ToolManager:
    def __init__(self, llm, data_dict):
        self.llm = llm
        self.data_dict = data_dict
        self.tools = []  # Initialize as an empty list

    def initialize_tool(self, tool_type, **kwargs):
        if tool_type == 'general_search':
            return self._initialize_search_tool()
        elif tool_type == 'financial_search':
            return self._initialize_financial_search_tool()
        elif tool_type == 'pandas':
            return self._initialize_pandas_tool(kwargs.get('df_name'), kwargs.get('tool_description'))
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
    
    def _initialize_search_tool(self):
        
        search = SerpAPIWrapper()

        self.add_tool(
            name="General Search",
            func=search.run,
            description="Useful for answering questions about current events. You should ask targeted questions."
        )

    def _initialize_financial_search_tool(self):
        
        search = SerpAPIWrapper()

        self.add_tool(
            name="Financial Search",
            func=search.run,
            description="Useful for answering questions about stock prices, market capitalization, company performances, interest rates, etc. When searching for stock prices, look into answer_box to get the current price and / or other information. You should ask targeted questions."
        )

    def _initialize_pandas_tool(self, df_name, tool_description):
        # Loading data
        
        excel_dict = self.data_dict.get('excel', None) 

        if excel_dict:
            data = excel_dict.get(df_name, None)

            if data is not None:
                if isinstance(data, dict):  # Multiple sheets/dataframes
                    list_of_dfs = list(data.values())
                    descriptions = []

                    # Iterate through sheets and their columns to construct the description
                    for index, (sheet_name, df) in enumerate(data.items(), start=1):
                        columns = ", ".join(df.columns.values)
                        descriptions.append(f"Sheet {index}, {sheet_name}, with the columns: {columns}")

                    # Join the individual descriptions
                    combined_description = ", ".join(descriptions)

                    df_pandas_agent = create_pandas_dataframe_agent(self.llm, list_of_dfs, verbose=True, max_iterations=5)
                    tool_name = f"{df_name.capitalize()} Pandas Agent (Multiple Sheets)"
                    self.add_tool(
                        name=tool_name,
                        func=df_pandas_agent.run,
                        description=f"{tool_description}. The file contains {combined_description}."
                    )
                else:  # Single dataframe
                    if not data.empty:

                        df_columns = data.columns.values
                        df_pandas_agent = create_pandas_dataframe_agent(self.llm, data, verbose=True, max_iterations=5)
                        tool_name = f"{df_name.capitalize()} Pandas Agent"
                        self.add_tool(
                            name=tool_name,
                            func=df_pandas_agent.run,
                            description=f"{tool_description}. The file contains the columns: {df_columns}."
                        )
                return df_pandas_agent

    def initialize_conversational_qa_tool(self, conversational_qa_agent, vectorstore_name, tool_description):
        # import chain and agent 
        tool_name = f"{vectorstore_name.capitalize()} Question Retrieval Agent"
        self.add_tool(
            name=tool_name,
            func=conversational_qa_agent.run,
            description=f"{tool_description}"
        )
        