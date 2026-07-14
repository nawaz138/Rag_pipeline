import os 
os.environ["OPENAI_API_KEY"]='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'
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model = "gpt-5.1")

from profileschema import result,candidate_projects,candidate_skills

JD= '''
We are looking for a passionate Generative AI Engineer to design, develop, and deploy AI-powered applications using Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), AI agents, and modern AI frameworks. The ideal candidate should have hands-on experience in Python, GenAI frameworks, prompt engineering, vector databases, and cloud AI platforms.

Key Responsibilities
Design and build end-to-end Generative AI solutions for enterprise use cases.
Develop RAG-based applications using vector databases and embedding models.
Build AI Agents using frameworks such as LangChain, LangGraph, CrewAI, or similar.


ChromaDB
Pinecone
Milvus
Weaviate
Azure AI Search
LLM Platforms

Experience with one or more:

OpenAI GPT
Azure OpenAI
Anthropic Claude
oRA)
Knowledge Graphs
GraphRAG
MLOps
MLflow
LangSmith
Prompt evaluation frameworks
AI observability tools
Qualifications

Hugging Face profile
AI demos or personal projects
Technical blogs or publications
Contributions to open-source AI projects
'''

from typing import Literal
class ResultSchema(BaseModel):
  '''
  profile schema
  '''
  Result: Literal["Selected", "Not Selected"] = Field(description="result of the selection")
  Reason: str = Field(description="reason for selection")

from langchain.agents import create_agent
selection_agent = create_agent(
    model = llm,
    tools = [],
    system_prompt = '''
                    you are expert in analyzing Candidate skill and project and matching with JD mentioned bby HR in order to select or reject candidate
                    ''',
    response_format=ResultSchema,

)

user_input = f'''
              candidate_skills: {candidate_skills}
              candidate_projects: {candidate_projects}
              JD: {JD}
              '''
final_response = selection_agent.invoke({"messages":[{"role":"user","content":user_input}]})

print(final_response["structured_response"].Result)
print(final_response["structured_response"].Reason)