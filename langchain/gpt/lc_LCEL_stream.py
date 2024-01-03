from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

prompt = ChatPromptTemplate.from_template("Tell me a short joke about {topic}")
output_parser = StrOutputParser()
model = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key="",
)
chain = {"topic": RunnablePassthrough()} | prompt | model | output_parser

print(chain.invoke("ice cream"))

for chunk in chain.stream("ice cream"):
    print(chunk, end="", flush=True)
