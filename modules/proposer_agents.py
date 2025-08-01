

# Define an agent class with name, type, and description
class Agent:
    def __init__(self, name, tag_name, agent_type, llm, description):
        self.name = name
        self.tag_name = tag_name
        self.agent_type = agent_type
        self.llm = llm
        self.description = description

    def __str__(self):
        return f"Name: {self.name}\nTag name: {self.tag_name}\nType: {self.agent_type}\nLLM: {self.llm}\nDescription: {self.description}"

    def __repr__(self):
        return f"Agent(name={self.name}, tag_name={self.tag_name}, agent_type={self.agent_type}, llm={self.llm}, description={self.description})"


# Get the agents from agent df
def get_proposer_and_aggregator_agents(agents_df):
  """
  Get the proposer and aggregator agents from the agents dataframe.
  Args:
    agents_df: The dataframe containing the agents.
  Returns:
    proposer_agents: A list of proposer agents.
    aggregator_agent: The aggregator agent.
  """
  proposer_agents = []
  aggregator_agent = ''

  for ind, row in agents_df.iterrows():
      if row['Agent Type'] == 'Proposer':
          proposer_agents.append(Agent(row['Name'], row['tag_name'], row['Agent Type'], row['LLM'], row['Description']))
      elif row['Agent Type'] == 'Aggregator':
          aggregator_agent = Agent(row['Name'], row['tag_name'], row['Agent Type'], row['LLM'], row['Description'])

  return proposer_agents, aggregator_agent
