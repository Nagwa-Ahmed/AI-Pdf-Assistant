import streamlit as st
from llama_index.llms.ollama import Ollama
from pypdf import PdfReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core import Document

# Initialize embeddings model
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# Initialize language model
Settings.llm = Ollama(model="llama3", request_timeout=1000)
#Setting the chatbot title and an image
st.title("💬 AI Pdf Assistant")

# In case that the session state has no messages(i.e., no prompts) ==> show How can I help you
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# Write Message History
#Each time the app responds , it writes the whole prompts and responses. First==> user's prompt, second==> model's response
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message(msg["role"], avatar="🧑‍💻").write(msg["content"])
    else:
        st.chat_message(msg["role"], avatar="🤖").write(msg["content"])

# File uploader for PDF
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
# in case the user added a file
if uploaded_file is not None:
    # Read the PDF file
    pdf_reader = PdfReader(uploaded_file)
    # Initialize the pdf_text variable to append file contents
    pdf_text = ""
    #Iterate over each page
    for page_num in range(len(pdf_reader.pages)):
        # Extract its contents and add it to pdf_text
        page = pdf_reader.pages[page_num]
        pdf_text += page.extract_text()

    # Convert PDF text to a Document object, to be saved in the vector store index
    document = Document(text=pdf_text)
# In case, the user provide a prompt then check if there is an uploaded file
# Else don't do any thing
if prompt := st.chat_input():
    # In case there is a provided pdf , do the following
    # Else raise an error (Upload a pdf file please)
    if uploaded_file is not None:
        # Create index from the document
        index = VectorStoreIndex.from_documents([document])
        #Setting the query engine
        query_engine = index.as_query_engine(streaming=True)# here we specify to have a streamed response

        # Perform RAG query
        response = query_engine.query(prompt)
        # add the prompt and response to messages in session state
        st.session_state.messages.append({"role": "user", "content": prompt})

        st.session_state.messages.append({"role": "assistant", "content": response.response_gen})# in order to have an iterable object to be streamed we use the attribute "response_gen", to iterate over tokens

        # Display the new user message
        st.chat_message("user", avatar="🧑‍💻").write(prompt)
        # Display the assistant response
        st.chat_message("assistant", avatar="🤖").write(response.response_gen)
    else:
        st.error("Upload a pdf file please")
else:
    pass
