from dotenv import load_dotenv
import os
load_dotenv()
from crewai import LLM , Agent,Task,Crew
llm  = LLM(
    model="llama-3.3-70b-versatile",
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY"),
    temperature = 0.1

)
email_assistant=Agent(
    role = "email_assistant",
    goal="Improve emails and make them sound professional and clear",
    backstory="A highly experienced communication expert  skilled in professional email writing and editing. Known for crafting clear, concise, and impactful messages that resonate with diverse audiences.",
    verbose = True,
    llm=llm
)
original_email = """
hey team, just wanted to tell you that demo  is kind of ready,but there's still stuff left.\
Maybe we can show what we have and say rest is WIP.
Let me know what u think. thanks
 """
email_task=Task(
    description=f"""Take the following rough email and rewrite it into a professional and polished version
    Expand abbreviations:
    '''{original_email}'''""",
    agent = email_assistant,
    expected_output="A professional written email with proper formatting and content.",
)
crew = Crew(
    agents=[email_assistant],
    tasks = [email_task],
    verbose = True
)
result = crew.kickoff()
print(result)
