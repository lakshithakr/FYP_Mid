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
    html_temp = """
    <div style="background-color: #33acff; padding: 15px; margin-bottom: 15px; border-radius: 10px; box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);">
        <h1 style="color: #000000; text-align: center; font-family: Arial, sans-serif;">Dominious</h1>
        <h2 style="color: #ffffff; text-align: center; font-family: Georgia, serif; font-size: 22px;">AI Based Domain Name Suggestion System</h2>
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
            st.success(domain_names[i])

if __name__ == "__main__":
    main()