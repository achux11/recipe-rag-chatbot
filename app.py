import streamlit as st

from rag import generate_answer


st.set_page_config(page_title="Recipe RAG Assistant", layout="wide")
st.title("🍲 Recipe Recommendation RAG Assistant")

st.markdown(
    """
**Welcome to your own recipe treasure.**

This assistant uses your own recipe PDFs / text files ingested into a vector database.
"""
)


if "history" not in st.session_state:
    st.session_state.history = []  # list of {user: str, assistant: str}


with st.sidebar:
    st.header("Controls")
    if st.button("Clear conversation"):
        st.session_state.history = []
    st.markdown(
        "Make sure you've:\n"
        "1. Added recipe PDFs / .txt files into the `data` folder\n"
        "2. Run `python ingest.py` before starting this app"
    )


# Render previous conversation
for turn in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(turn["user"])
    with st.chat_message("assistant"):
        st.markdown(turn["assistant"])


user_input = st.chat_input("Describe ingredients or ask for a recipe...")

if user_input:
    # Show user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate assistant response with RAG
    with st.chat_message("assistant"):
        with st.spinner("Thinking about recipes..."):
            answer = generate_answer(user_input, st.session_state.history)
            st.markdown(answer)

    # Update conversation history (memory)
    st.session_state.history.append({"user": user_input, "assistant": answer})


