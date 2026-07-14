import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI


load_dotenv()


api_key = os.getenv(
    "OPENAI_API_KEY"
)





llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    api_key=api_key
)

from pydantic import BaseModel, Field
class ProfileSchema(BaseModel):

    name: str = Field(
        description="candidate name"
    )

    email: str = Field(
        description="candidate email"
    )

    skills: list = Field(
        description="candidate skills"
    )

    experience: str = Field(
        description="candidate experience"
    )

    projects: str = Field(
        description="candidate projects"
    )

from langchain.agents import create_agent
profile_agent = create_agent(
    model=llm,
    tools=[],
    system_prompt="""

You are an expert resume analyzer.

Extract candidate information from resume.

Return:
name
email
skills
experience
projects

""",
    response_format=ProfileSchema
)



def analyze_resume(text):

    response = profile_agent.invoke(
        {
            "messages":[
                {
                    "role":"user",
                    "content":text
                }
            ]
        }
    )


    return response["structured_response"]