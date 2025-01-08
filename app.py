from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import streamlit as st

st.title("LLama 3.1 ChatBot")

# Styling
st.markdown(
    """
    <style>
    .main {
        background-color: #00000;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar for additional options or information
with st.sidebar:
    st.info("This app uses the Llama 3.1 model to answer your questions.")

# Initialize session state for storing past responses
if "history" not in st.session_state:
    st.session_state["history"] = []

# Chat prompt and model setup
template = """Question: {question}
Answer: Let's think step by step."""
prompt = ChatPromptTemplate.from_template(template)
model = OllamaLLM(model="llama3.1")
chain = prompt | model

# Main content
col1, col2 = st.columns(2)
with col1:
    question = st.text_input("Enter your question here")

# Display previous responses
st.markdown("### Chat History")
if st.session_state["history"]:
    for i, chat in enumerate(st.session_state["history"], 1):
        st.markdown(f"**Q{i}:** {chat['question']}")
        st.markdown(f"**A{i}:** {chat['answer']}")

if question:
    with st.spinner("Thinking..."):
        answer = chain.invoke({"question": question})
        st.success("Done!")
    
    # Save the question and answer to the session state history
    st.session_state["history"].append({"question": question, "answer": answer})
    
    # Display the latest answer
    st.markdown(f"**Answer:** {answer}")
else:
    st.warning("Please enter a question to get an answer.")
