from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


embeddings = OllamaEmbeddings(model="nomic-embed-text")


db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)


retriever = db.as_retriever(search_kwargs={"k": 3})
                            
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a corporate SOP assistant.
Answer ONLY from the given context.
If the answer is not present, say:
"I don't know. This information is not available in the documents."

Context:


Question:
{question}

Answer:
"""
)


llm = Ollama(model="llama2")


qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
)


while True:
    query = input("\nAsk a question (or type 'exit'): ")
    if query.lower() == "exit":
        break

    result = qa.run(query)
    print("\nAnswer:", result)