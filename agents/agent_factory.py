# agents/agent_factory.py

from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from utils.pinecone_utils import init_langchain_clients, cargar_embeddings_si_no_existen, cargar_embeddings_mixer_si_no_existen
from utils.common import CustomAzureChatModel

def crear_agente_cv(index_name, system_prompt, cv_file=None, source=None, is_mixer=False):
    vectorstore = init_langchain_clients(index_name)
    
    if is_mixer:
        cargar_embeddings_mixer_si_no_existen(vectorstore)
    elif cv_file and source:
        cargar_embeddings_si_no_existen(vectorstore, cv_file, source)
    
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5, "lambda_mult": 0.25})
    llm = CustomAzureChatModel(model="gpt-4o-mini", temperature=0.5)
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        MessagesPlaceholder(variable_name="historial_chat"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])
    combine_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    return create_retrieval_chain(retriever=retriever, combine_docs_chain=combine_chain)
