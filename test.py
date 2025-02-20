import streamlit as st
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import Html2TextTransformer


def main():
    st.title("Test")

    st.header("test")

    url = st.text_input("Enter url: ")

    if st.button("Load web page text"):
        loader = AsyncChromiumLoader([url]) # Only pass a single url
        docs = loader.load()
        html2text = Html2TextTransformer(ignore_links = False, ignore_images = False)
        doc = html2text.transform_documents(docs)[0]  # Return only the first (only) element

        st.write(doc.metadata)
        st.write(doc.page_content)
    
    st.write("END")


if __name__ == "__main__":
    main()
