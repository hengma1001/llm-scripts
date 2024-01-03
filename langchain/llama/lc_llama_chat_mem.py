from os.path import expanduser

from langchain.chains import LLMChain
from langchain.llms import LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import SystemMessage
from langchain_experimental.chat_models import Llama2Chat

template_messages = [
    SystemMessage(content="You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{text}"),
]
prompt_template = ChatPromptTemplate.from_messages(template_messages)


model_path = expanduser(
    "/homes/heng.ma/Research/md_pkgs/llm-scripts/langchain/llama/llama-2-7b-chat.Q4_0.gguf"
)

n_gpu_layers = 40  # Change this value based on your model and your GPU VRAM pool.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

llm = LlamaCpp(
    model_path=model_path,
    # max_tokens=5000,
    # n_gpu_layers=n_gpu_layers,
    # n_batch=n_batch,
    streaming=False,
    Verbose=True,
)
model = Llama2Chat(llm=llm)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
chain = LLMChain(llm=model, prompt=prompt_template, memory=memory)

try:
    ind = 0
    while True:
        print(f"Q{ind}========================")
        question = input("Question: ")
        # print("Question: " + question)
        print("Answer: " + chain.run(text=question))
        ind += 1

except KeyboardInterrupt:
    print("Done")
# print("=======================")
# print(
#     chain.run(
#         text="What can I see in Vienna? Propose a few locations. Names only, no details."
#     )
# )
# print("=======================")
# print(chain.run(text="Tell me more about #3."))
