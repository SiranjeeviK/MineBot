import streamlit as st
import openai
from dotenv import load_dotenv
# from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
# from htmlTemplates import bot_template, user_template, CSS

#############################
CSS ='''
<style>
    .chat-message {
        padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex;
    }
    .chat-message.user {
        background-color: #2b313e;
    }
    .chat-message.bot {
        background-color: #475063;
    }
    .chat-message .avatar {
        width: 15%;
    }

    .chat-message .avatar img {
        max-width: 78px;
        max-height: 78px;
        border-radius: 50%;
        object-fit: contain;
    }

    .chat-message .message {
        width: 85%;
        padding: 0 1.5rem;
        color: #fff;
    }
        '''

bot_template = '''

<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.ibb.co/vvPvZcv/robot-croped.gif" alt="Bot">
    </div>
    <div class="message">{{MSG}} </div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://i.ibb.co/c8s2Mmb/image.png" alt="User">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''
#############################

from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper

from langchain.utilities import SerpAPIWrapper

from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory

from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import BaseChatPromptTemplate
from langchain.utilities import SerpAPIWrapper
from langchain.chains.llm import LLMChain
from langchain.chat_models import ChatOpenAI
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, HumanMessage
import re

started = False

def get_vectorstore():
    # from langchain.embeddings import HuggingFaceInstructEmbeddings
    # from langchain.vectorstores import FAISS

    # pinecone.create_index("langchain-minebot", dimension=768)
    # embeddings = HuggingFaceInstructEmbeddings(model_name = 'hkunlp/instructor-xl')
    embeddingsBgeLarge = HuggingFaceInstructEmbeddings(model_name = 'BAAI/bge-large-en-v1.5', cache_folder ="model_cache")
    print("The embedding Model is loaded")
    vectorstore = FAISS.load_local("clean_total_index_bge_large", embeddingsBgeLarge)
    print("The vectorstore is loaded")
    return vectorstore

def get_tools(db):
    # from langchain.tools import WikipediaQueryRun
    # from langchain.utilities import WikipediaAPIWrapper
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

    # from langchain.utilities import SerpAPIWrapper
    search = SerpAPIWrapper()

    # from langchain.chains import RetrievalQA
    llm = OpenAI(model="gpt-3.5-turbo-instruct")
    db_retriever = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=db.as_retriever()
    )
    tools = [
        Tool(
            name="Mining related database",
            func=db_retriever.run,
            description="useful for when you need to answer questions about Mining related questions such as The legal provisions encompass amendments and regulations related to coal and mineral mining in India. They include updates to colliery control, coal block allocation, and mineral concession rules. Additionally, acts like The Coal Mines (Special Provisions) Act, 2015, and The Mineral Laws (AMENDMENT) ACT, 2020, are integral to governing mining activities and land use in the country.",
        ),

         Tool(
            name="Search",
            func=search.run,
            description="Useful when you need to answer questions about current events. You should ask targeted questions.",
        ),

        Tool(
            name="Wikipedia",
            func=wikipedia.run,
            description="useful for when you need to use wikipedia, always give the source link at the end"
        )
    ]

    return tools


def get_memory():
    # from langchain.memory import ConversationBufferMemory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return memory


def get_agent(tools, memory):
    # from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
    # from langchain.prompts import BaseChatPromptTemplate
    # from langchain.utilities import SerpAPIWrapper
    # from langchain.chains.llm import LLMChain
    # from langchain.chat_models import ChatOpenAI
    # from typing import List, Union
    # from langchain.schema import AgentAction, AgentFinish, HumanMessage
    # import re

    # Set up the base template
    template = """Complete the objective as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    These were previous tasks you completed:



    Begin!
    {chat_history}
    Question: {input}
    {agent_scratchpad}"""


    # Set up a prompt template
    class CustomPromptTemplate(BaseChatPromptTemplate):
        # The template to use
        template: str
        # The list of tools available
        tools: List[Tool]

        def format_messages(self, **kwargs) -> str:
            # Get the intermediate steps (AgentAction, Observation tuples)
            # Format them in a particular way
            intermediate_steps = kwargs.pop("intermediate_steps")
            thoughts = ""
            for action, observation in intermediate_steps:
                thoughts += action.log
                thoughts += f"\nObservation: {observation}\nThought: "
            # Set the agent_scratchpad variable to that value
            kwargs["agent_scratchpad"] = thoughts
            # Create a tools variable from the list of tools provided
            kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
            # Create a list of tool names for the tools provided
            kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
            formatted = self.template.format(**kwargs)
            return [HumanMessage(content=formatted)]




    prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "chat_history" ,"intermediate_steps"]
    )

    class CustomOutputParser(AgentOutputParser):

        def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
            # Check if agent should finish
            if "Final Answer:" in llm_output:
                return AgentFinish(
                    # Return values is generally always a dictionary with a single `output` key
                    # It is not recommended to try anything else at the moment :)
                    return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                    log=llm_output,
                )
            # Parse out the action and action input
            regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
            match = re.search(regex, llm_output, re.DOTALL)
            if not match:
                raise ValueError(f"Could not parse LLM output: `{llm_output}`")
            action = match.group(1).strip()
            action_input = match.group(2)
            # Return the action and action input
            return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

    output_parser = CustomOutputParser()
    llm = ChatOpenAI(temperature=0)
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\nObservation:"],
    allowed_tools=tools
    )
    print("Tools Ready")
    return agent




def get_conversation_chain(agent, tools, memory):

    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        memory = memory
        )
    print("Agent Executor ready")
    return agent_executor


def handle_userinput(user_question):
    response = st.session_state.conversation(user_question)
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i%2 == 0:
            st.write(user_template.replace('{{MSG}}', message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace('{{MSG}}', message.content), unsafe_allow_html=True)

    # st.chat_input('Ask a question')

def main():
    load_dotenv()
    # 🤑
    st.set_page_config(page_title='MineBot', page_icon=':money-month face:')
    st.write(CSS, unsafe_allow_html=True)

    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = None

    st.header('Minebot')
    user_question = st.chat_input('Ask a question')
    if user_question:
        global started
        started = True
        with st.spinner(user_question):
            handle_userinput(user_question)
    # st.write(user_template, unsafe_allow_html=True)
    # st.write(bot_template, unsafe_allow_html=True)

    with st.sidebar:
        # st.subheader("Your Documents")
        # pdf_docs = st.file_uploader("Upload the PDF file and click 'Process'", accept_multiple_files=True)

        # Give a space system vibe for activation the procees include loading embbedding model, loading FAISS local index for vectorstore, loading tools like wikipedia api, serp search api, then custom prompt templete, memory, prompt, output parser, agent, finally agent executor
        st.header("MineBot")
        st.subheader("Activate System")

        if st.button('Activate'):
            with st.spinner('Activating...'):

                # create a vector store
                vectorstore = get_vectorstore()
                st.session_state.vectorstore = vectorstore
                print("The vectorstore is created")
                print(vectorstore)
                st.write("Vector Store Ready!")

                tools = get_tools(vectorstore)
                st.write("Tools Ready!")

                memory = get_memory()
                st.write("Memory Ready!")
                agent = get_agent(tools, memory)

                # Create a conversation chain
                st.session_state.conversation = get_conversation_chain(agent, tools, memory)
                st.write("System is ready to answer your questions")
                print("Start the Music")
                
    # global started            
    if(started is False):         
        st.write(bot_template.replace('{{MSG}}', """
Welcome to MineBot!

👷‍♂️🚀 Hello there! I'm MineBot, your go-to assistant for all things related to mining laws and regulations. Whether you have questions about Acts, Rules, DGMS Circulars, or any mining-related topic, I'm here to help.

How can I assist you today?

Feel free to ask any questions, seek guidance, or stay updated on the latest mining industry information. I'm available 24/7 to provide you with accurate and timely responses.

Let's make navigating through mining regulations a breeze! What can I do for you?
"""), unsafe_allow_html=True)





if __name__ == '__main__':
    main()
    # used to run the app, if you are using streamlit run app.py, you don't need this line
