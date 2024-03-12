import langchain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain

import dotenv

dotenv.load_dotenv()
demo_chat_history = ChatMessageHistory()
langchain.verbose = False

chat = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.2)

SYSTEM_TEMPLATE = """
Answer the user's question based on the below context.
If the context doesn't contain any relevant information to the question, don't make something up and just say "I don't know"

<context>
{context}
</context>
"""

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            SYSTEM_TEMPLATE
        ),
        MessagesPlaceholder(variable_name="messages")
    ]
)

document_chain = create_stuff_documents_chain(chat, prompt)
loader = WebBaseLoader("https://finance.yahoo.com/")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=0) 
all_splits = text_splitter.split_documents(data)
vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever(k=4)
docs = retriever.invoke("provide a summary of this page")

print(docs)

close_keyword = "end"
print("Welcome to the chat bot demo!")
user_prompt = ""

while (user_prompt != close_keyword):
    user_prompt = input(">")
    if user_prompt == close_keyword:
        break 
    demo_chat_history.add_user_message(user_prompt)
    response = document_chain.invoke(
       {"messages": demo_chat_history.messages,
        "context": retriever.invoke(user_prompt)} 
    )
    demo_chat_history.add_ai_message(response)
    print(response)