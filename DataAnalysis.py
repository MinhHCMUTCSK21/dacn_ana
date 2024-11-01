import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI

from src.logger.base import BaseLogger
from src.models.llms import load_llm
from src.utils import execute_plt_code

# Load environment variables
load_dotenv()
logger = BaseLogger()
MODEL_NAME = "gpt-3.5-turbo"

def process_query(da_agent, query):
    response = da_agent(query)
    action = response["intermediate_steps"][-1][0].tool_input["query"]

    if "plt" in action:
        st.write(response["output"])
        fig = execute_plt_code(action, df=st.session_state.df)
        if fig:
            st.pyplot(fig)
        st.write("**Executed code:**")
        st.code(action)
        to_display_string = response["output"] + "\n" + f"```python\n{action}\n```"
        st.session_state.history.append((query, to_display_string))
    else:
        st.write(response["output"])
        st.session_state.history.append((query, response["output"]))

def display_chat_history():
    st.markdown("## Chat History: ")
    for i, (q, r) in enumerate(st.session_state.history):
        st.markdown(f"**Query: {i+1}:** {q}")
        st.markdown(f"**Response: {i+1}:** {r}")
        st.markdown("---")

def main():
    # Set up Streamlit interface
    st.set_page_config(page_title="Data Analysis Tool", page_icon="📊", layout="centered")
    st.header("Data Analysis Tool")
    st.write("### Welcome to the data analysis tool. This tool can assist with data analysis tasks for Bkareer. Please enjoy!")

    # Load LLM model
    llm = load_llm(model_name=MODEL_NAME)
    logger.info(f"### Successfully loaded {MODEL_NAME} !###")

    # Sidebar options
    with st.sidebar:
        option = st.selectbox("Choose how to load data:", ["Upload CSV File (Under 200MB)", "Use our survey"])
        
        if option == "Upload CSV File (Under 200MB)":
            uploaded_file = st.file_uploader("Upload your CSV file here", type="csv")
            st.session_state.df = None
        else:
            # Use a predefined DataFrame (example data)
            uploaded_file = None
            st.session_state.df = pd.read_csv("data.csv")
            st.write("### Our survey data:", st.session_state.df.head())

    # Initial chat history
    if "history" not in st.session_state:
        st.session_state.history = []

    # Read CSV file
    if uploaded_file is not None:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.write("### Your uploaded data: ", st.session_state.df.head())

    if st.session_state.df is not None:
        # Create data analysis agent to query with our data
        da_agent = create_pandas_dataframe_agent(
            llm=llm,
            df=st.session_state.df,
            agent_type="tool-calling",
            allow_dangerous_code=True,
            verbose=True,
            return_intermediate_steps=True,
        )
        logger.info("### Successfully loaded data analysis agent !###")

        # Input query and process query
        query = st.text_input("Enter your questions: ")

        if st.button("Run query"):
            with st.spinner("Processing..."):
                process_query(da_agent, query)
    
    # Display chat history
    st.divider()
    display_chat_history()

if __name__ == "__main__":
    main()

