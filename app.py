import streamlit as st
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os


st.title('AI IRISH LAWYER ')

load_dotenv()  # Load variables from .env file

@st.cache(allow_output_mutation=True)
def initialize_models():
    loader = UnstructuredPDFLoader("Law.pdf")
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(data)

     # Retrieve API keys from environment variables
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    pinecone.init(api_key=pinecone_api_key, environment="gcp-starter")
    index_name = "data"

    docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key,temperature=0.2, max_tokens=1000)

    template = "You are Irish law expert. Answer in detail in the language the question was asked. {documents}"
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    chain = LLMChain(llm=llm, prompt=chat_prompt)

    return docsearch, chain

docsearch, chain = initialize_models()

user_query = st.text_input("Enter your question:")

if st.button("Ask"):
    docs = docsearch.similarity_search(user_query)
    input_data = {
        'documents': docs,
        'question': user_query
    }
    result = chain.run(**input_data)

    st.write(f"User Query: {user_query}")
    st.write(f"Response: {result}")