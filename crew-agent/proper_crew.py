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
research_agent=Agent(
    role="Research speciliast",
    goal="Research facts about the topic: {topic}",
    backstory="You are expert at finding relevant and useful data.",
    verbose=True,
    llm=llm

)

writer_agent = Agent(
    role="creative writer",
    goal="write a short blog summary using the research",
    backstory="you are skilled at writing engaging summaries  based on provided content.",
    llm=llm,
    verbose=True,

)

task1 = Task(
    description="find 3-5 interesting and recent facts about {topic}.",
    expected_output = "a bullet list of 3-5 facts",
    agent = research_agent,
)
task2 = Task(
    description="write a short blog summary about {topic}using facts from  research..",
    expected_output = "a short blog summary",
    agent = writer_agent,
    context=[task1],
)

crew = Crew(
    agents=[research_agent, writer_agent],
    tasks=[task1, task2],
    verbose=True,
)

crew.kickoff(inputs={"topic":"ai in healthcare"})