"""Streamlit chat front-end for the Egypt Travel Guide RAG assistant."""
import streamlit as st

from api_client import APIError, ApiClient

st.set_page_config(page_title="Egypt Travel Guide Assistant", page_icon="🏺", layout="centered")

EXAMPLES = [
    "How much is a visitor ticket to the Temple of Edfu?",
    "What are the opening hours of the Nubian Museum?",
    "What is the distance between Siwa Oasis and Alexandria?",
    "What is the average maximum temperature in Aswan in August?",
    "Who founded Alexandria and when?",
    "What are the emergency phone numbers in Egypt?",
]

if "messages" not in st.session_state:
    st.session_state.messages = []      # [{"role": "user"|"assistant", "content": str, "sources": [str]}]


def get_client():
    try:
        return ApiClient(), None
    except APIError as exc:
        return None, str(exc)


client, config_error = get_client()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("🏺 About")
    st.write("Ask questions about travel in Egypt. Answers are generated **only** from four official guides "
             "(Egypt Tourism Guide, Alexandria, Egypt Map & Guide, Aswan Museums & Sites) and always show their sources.")
    st.subheader("Backend status")
    if config_error:
        st.error(config_error)
    else:
        try:
            h = client.health()
            if h.get("status") == "ok":
                st.success(f"Ready · {h.get('chunks', 0)} passages · model {h.get('llm_model')}")
            else:
                st.warning("Backend is up but not fully ready: "
                           + ("vector store not loaded. " if not h.get("vector_store_loaded") else "")
                           + ("Ollama is not reachable." if not h.get("ollama_reachable") else ""))
        except APIError as exc:
            st.error(str(exc))
    st.subheader("Try an example")
    for i, ex in enumerate(EXAMPLES):
        if st.button(ex, key=f"ex{i}", use_container_width=True):
            st.session_state.pending = ex
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------- chat
st.title("🏺 Egypt Travel Guide Assistant")
st.caption("Grounded answers with cited sources · powered by a local LLM (Ollama) and a Chroma vector store")


def render(msg):
    with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🏺"):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.markdown("**Sources**")
            for n, s in enumerate(msg["sources"], 1):
                st.markdown(f"<span style='font-size:0.9em'>[{n}] {s}</span>", unsafe_allow_html=True)


if not st.session_state.messages:
    st.info("👋 Ask me anything about Egypt — ticket prices, opening hours, distances, temperatures, history… "
            "Pick an example in the sidebar or type below.")
for m in st.session_state.messages:
    render(m)

question = st.chat_input("Ask a question about Egypt…", disabled=client is None)
question = question or st.session_state.pop("pending", None)

if question and client:
    user_msg = {"role": "user", "content": question, "sources": []}
    st.session_state.messages.append(user_msg)
    render(user_msg)
    with st.chat_message("assistant", avatar="🏺"):
        try:
            with st.spinner("Searching the guides and writing the answer…"):
                result = client.ask(question)
            reply = {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
        except APIError as exc:
            reply = {"role": "assistant", "content": f"⚠️ {exc}", "sources": []}
    st.session_state.messages.append(reply)
    st.rerun()      # re-render the whole history (keeps the layout consistent)
