import os
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_google_vertexai import VertexAI 
import networkx as nx
from langchain.chains import GraphQAChain
from langchain_core.documents import Document
from langchain_community.graphs.networkx_graph import NetworkxEntityGraph
import openai
from langchain_openai import ChatOpenAI
# 原来是这样call的 但是deepseek api key在这里用不了
# 我试了下面这个但是“openai.UnprocessableEntityError: Failed to deserialize the JSON body into the target type: response_format: response_format.type `json_schema` is unavailable now at line 1 column 5369”
'''llm = ChatOpenAI(
    model_name="gpt-4",
    openai_api_key="sk“
    '''
llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key="sk-", 
    openai_api_base="https://api.deepseek.com/v1"  
)


text = """
Marie Curie, born in 1867, was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity.
She was the first woman to win a Nobel Prize, the first person to win a Nobel Prize twice, and the only person to win a Nobel Prize in two scientific fields.
Her husband, Pierre Curie, was a co-winner of her first Nobel Prize, making them the first-ever married couple to win the Nobel Prize and launching the Curie family legacy of five Nobel Prizes.
She was, in 1906, the first woman to become a professor at the University of Paris. 
"""

documents = [Document(page_content=text)]
llm_transformer = LLMGraphTransformer(llm=llm)
graph_documents = llm_transformer.convert_to_graph_documents(documents)

llm_transformer_filtered = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=["Person", "Country", "Organization"],
    allowed_relationships=["NATIONALITY", "LOCATED_IN", "WORKED_AT", "SPOUSE"],
)
graph_documents_filtered = llm_transformer_filtered.convert_to_graph_documents(
    documents
)

graph = NetworkxEntityGraph()

# Add nodes to the graph
for node in graph_documents_filtered[0].nodes:
    graph.add_node(node.id)

# Add edges to the graph
for edge in graph_documents_filtered[0].relationships:
    graph._graph.add_edge(
            edge.source.id,
            edge.target.id,
            relation=edge.type,
        )

chain = GraphQAChain.from_llm(
    llm=llm, 
    graph=graph, 
    verbose=True
)

question = """Who is Marie Curie?"""
answer = chain.run(question)
print(answer)