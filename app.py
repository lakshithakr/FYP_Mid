import streamlit as st
from langchain import HuggingFaceHub
from apikey_hungingface import apikey_hungingface
from langchain import PromptTemplate, LLMChain
import os
import re

# Set Hugging Face Hub API token
# Make sure to store your API token in the `apikey_hungingface.py` file
os.environ["HUGGINGFACEHUB_API_TOKEN"] = apikey_hungingface

# Set up the language model using the Hugging Face Hub repository
repo_id = "tiiuae/falcon-7b-instruct"
llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.8, "max_new_tokens": 2048})

# Set up the prompt template
template = """
Context: {context}
Task: Based on the context above, suggest ten unique and creative names. The names should be aligned with the context and reflect the key themes or values. Ensure that the names are distinct, concise, and preferably short (one or two words). Do not repeat names and avoid generating similar names multiple times.
Note: Generate only one occurrence of each name. Ensure no repetition of names in the output.\n
Names:
"""
prompt = PromptTemplate(template=template, input_variables=["context"])
llm_chain = LLMChain(prompt=prompt, llm=llm)




# Create the Streamlit app
def main():

    css_dark_mode = """
    <style>
        body {
            background-color: #121212;
            color: #f8f9fa;
            font-family: Arial, sans-serif;
        }
        .header {
            background: linear-gradient(90deg, #005c97, #363795);
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 10px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
            text-align: center;
        }
        .header h1 {
            color: #ffffff;
            font-size: 36px;
        }
        .header h2 {
            color: #b0b0b0;
            font-size: 22px;
            font-family: Georgia, serif;
        }
        .stButton > button {
            background-color: #20c997;
            color: #ffffff;
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        .stButton > button:hover {
            background-color: #17a589;
        }
        .success-box {
            background-color: #A1D6CB;
            padding: 10px;
            margin: 10px 0;
            border-radius: 8px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
        }
    </style>
    """
    # Inject the custom CSS
    st.markdown(css_dark_mode, unsafe_allow_html=True)

    # Header Section
    html_temp = """
    <div class="header">
        <h1>Dominious</h1>
        <h2>AI Based Domain Name Suggestion System</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    # st.title("Domain Name Suggestion System")

    # Get user input
    context = st.text_input("Describe your business here...")

    # Generate the response
    if st.button("Generate"):
        with st.spinner("Generating Domain Names..."):
            result = llm_chain.run(context)
        lines = result.splitlines()
        start_index = lines.index("Names:") + 1
        domain_names = [line.strip() for line in lines[start_index:] if line.strip()]
        for i in range(len(domain_names)):
            st.markdown(f'<div class="success-box">{domain_names[i]}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()