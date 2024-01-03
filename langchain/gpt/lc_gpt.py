from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI

openai_api_key = ""

llm = OpenAI(openai_api_key=openai_api_key)
chat_model = ChatOpenAI(openai_api_key=openai_api_key)


from langchain.schema import HumanMessage

text = "What would be a good company name for a company that makes colorful socks?"
messages = [HumanMessage(content=text)]

print(llm.invoke(text))
# >> Feetful of Fun

print(chat_model.invoke(messages))
# >> AIMessage(content="Socks O'Color")
