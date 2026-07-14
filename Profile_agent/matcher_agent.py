from typing import Literal

from pydantic import BaseModel,Field
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent


import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI


load_dotenv()


llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
)



class ResultSchema(BaseModel):

    Result: Literal[
        "Selected",
        "Not Selected"
    ] = Field(
        description="candidate selection result"
    )


    Reason:str = Field(
        description="reason"
    )


from langchain.agents import create_agent
matcher_agent=create_agent(

    model=llm,

    tools=[],

    system_prompt="""

You are an expert technical recruiter.

Compare candidate skills and projects
with the Job Description.

Select candidate if suitable.

Explain reason.

""",

response_format=ResultSchema

)



def match_candidate(profile,jd):


    prompt=f"""

Candidate Skills:
{profile.skills}


Candidate Projects:
{profile.projects}


Experience:
{profile.experience}



Job Description:

{jd}

"""


    response=matcher_agent.invoke(
        {
            "messages":[
                {
                    "role":"user",
                    "content":prompt
                }
            ]
        }
    )


    return response["structured_response"]