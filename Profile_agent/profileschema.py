import os 
os.environ["OPENAI_API_KEY"]='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'

from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model = "gpt-5.1")
class ProfileSchema(BaseModel):
  '''
  profile schema
  '''
  name: str = Field(description="name of the candiate")
  email: str = Field(description="email id of the candidate")
  skills: list = Field(description="skills of the candidate")
  experience: str = Field(description="experience of the candidate")
  projects : str = Field(description="projects completed by the candidate")


from langchain.agents import create_agent
profile_extractor_agent = create_agent(
    model = llm,
    tools = [],
    system_prompt = '''
                    you are expert in analyzing candidate resume and extract things mentioned in shcema
                    ''',
    response_format=ProfileSchema,

)
profile = '''

Name of the Trainer: Harish Kumar
Experience Summary:
Lead Data Scientist with 12+ years of experience in Generative AI, NLP, Deep Learning and AI, Computer Vision, Data Visualization, Image Processing, Integration Development and Agile methodologies with loads of professional training to deliver highly effective and creative solutions to technology challenges with proven success in building successful algorithms & predictive models. Highly adept at solving process control tasks using Reinforcement Learning, Computer Vision, Image
Processing, data analysis & visualization to increase business efficiency. Passionate engineer & thriving analyst with the ability to apply ML techniques & algorithm development to solve real-world business problems. Experience in building Agentic AI solution using different LLM provider like Anthropic, AzureOpenAI with tools to build production ready workflows.

Technical Skills:
•	Generative AI Tools/Frameworks: ChatGPT, DALL·E, Copilot, Tabnine, LlamaIndex, LangChain, Prompt Engineering (Zero-shot, Few-shot, Prompt Chaining), AI Agents, Agentic AI, n8n, Vibe Coding, Anthropic Claude, Cohere, Retrieval-Augmented Generation (RAG), with awareness of AI limitations (bias, hallucination & data privacy)
•	Languages & Frameworks: Python, Java, SQL, Google Colab, Hibernate, Spring
•	Libraries & Packages: Scikit-Learn, XGBoost, NLTK, Keras, NumPy, SciPy, Pandas, BERT, OpenCV, Tesseract
•	MLOps Frameworks/Tools: MLflow, Airflow, Kubeflow, Databricks
•	DevOps Tools: Splunk, Jenkins, Postman, Docker, Kubernetes
•	IDEs: PyCharm, Eclipse, Anypoint Studio



Training Delivered / Expertise Areas
• Agentic AI for Solution Architects
• Agentic AI for Senior Data Scientists
• Generative AI Platforms (n8n, Julius, Zapier)
• Generative AI Training using Apple LLMs and Platforms
• Generative AI for Solution Architects
• Testing and Evaluation of AI Models
• Generative AI for FMCG Domain
• Generative AI for Data Scientists


Work Experience:


Brillio Technologies: Data Scientist
Project: Credit Card Fraud Detection |East West Bank
Tech Stack: Python, NumPy, Pandas, ML Algorithms, AWS (S3, Sagemaker, EC2), Scikit-Learn, Matplotlib, Keras, Tensorflow
•	Objective: To build models to detect fraud based on transactions history of customers
•	Solution: Handled imbalance class issue, missing values. created ML pipeline In AWS and Designed various ML models using KNN, Random Forest, XG Boost algorithms with PCA to identify fraud.
•	Key Achievement: Developed a model with an AUC score of 97%

Project: Customer Segmentation |East West Bank
Tech Stack: Python, NumPy, Pandas, ML Algorithm(K-Mean), AWS(S3,EC2,Sagemaker),pyspark Scikit-Learn, Matplotlib
•	Objective: To build customer segmentation to segment customer based on various details
•	Solution: Fetch data from S3 bucket, performed analysis using pyspark MLlib library and implemented K-means algorithm along with hyperparameter tuning.




'''

result = profile_extractor_agent.invoke({"messages":[{"role":"user","content":profile}]})
candidate_skills = result["structured_response"].skills
candidate_projects= result["structured_response"].projects

print(candidate_projects)
print(candidate_skills)

