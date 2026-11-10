import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.chains import (
    create_retrieval_chain,
    create_history_aware_retriever
)
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain
)



load_dotenv()

os.environ.setdefault(
    "USER_AGENT",
    "FastAPI-Docs-Chatbot/1.0"
)

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError(
        "GROQ_API_KEY not found. Check your .env file."
    )



print("\nInitializing Groq LLM...")


llm = ChatGroq(
    groq_api_key=groq_api_key,
    model="openai/gpt-oss-20b",
    max_tokens=1024,
    temperature=0.2
)


print("Groq LLM initialized!")




print("\nLoading FastAPI documentation...")

urls = [
    "https://fastapi.tiangolo.com/tutorial/first-steps/",
    "https://fastapi.tiangolo.com/tutorial/path-params/",
    "https://fastapi.tiangolo.com/tutorial/query-params/",
    "https://fastapi.tiangolo.com/tutorial/body/",
]

loader = WebBaseLoader(
    web_paths=tuple(urls)
)

docs = loader.load()

print("Total Documents Loaded:", len(docs))



print("\nSplitting documents...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

splits = text_splitter.split_documents(docs)

print("Total Chunks Created:", len(splits))



print("\nLoading HuggingFace embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded!")


print("\nConnecting to ChromaDB...")

vector_store = Chroma(
    collection_name="fastapi_docs",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)


if vector_store._collection.count() == 0:

    print("ChromaDB is empty. Adding documents...")

    vector_store.add_documents(splits)

    print("Documents added successfully!")

else:

    print(
        "Existing ChromaDB collection found:",
        vector_store._collection.count(),
        "chunks"
    )

print("ChromaDB ready!")


retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)

print("Retriever created!")



contextualize_q_prompt = ChatPromptTemplate.from_messages([

    (
        "system",
        """
        Given the conversation history and the latest user question,
        rewrite the latest question as a standalone question.

        Do not answer the question.

        If the latest question is already clear,
        return it without changing its meaning.
        """
    ),

    MessagesPlaceholder(
        variable_name="chat_history"
    ),

    ("human", "{input}")

])



history_aware_retriever = create_history_aware_retriever(
    llm,
    retriever,
    contextualize_q_prompt
)

print("History-aware retriever created!")



qa_prompt = ChatPromptTemplate.from_messages([

    (
        "system",
        """
        You are a helpful FastAPI documentation assistant.

        Your job is to answer questions about FastAPI.

        Follow these rules:

        1. Use the provided documentation context to answer.
        2. Use conversation history to understand follow-up questions.
        3. Explain concepts in simple language.
        4. Provide Python code examples when useful.
        5. Do not invent documentation facts.
        6. If the answer is not available in the context,
           clearly say that you don't know based on the
           available documentation.

        Documentation Context:

        {context}
        """
    ),

    MessagesPlaceholder(
        variable_name="chat_history"
    ),

    ("human", "{input}")

])

print("QA Prompt created!")



question_answer_chain = create_stuff_documents_chain(
    llm,
    qa_prompt
)

print("Question Answering Chain created!")




rag_chain = create_retrieval_chain(
    history_aware_retriever,
    question_answer_chain
)

print("Conversational RAG Chain created successfully!")


chat_history = []

print("\n========================================")
print("   FASTAPI CONVERSATIONAL RAG CHATBOT")
print("========================================")
print("Ask questions about FastAPI documentation.")
print("Type 'exit' to stop.")
print("Type 'clear' to clear conversation history.")
print("========================================\n")



while True:

    question = input("You: ").strip()

   
    if question.lower() in ["exit", "quit"]:
        print("\nChatbot stopped. Goodbye!")
        break

    if question.lower() == "clear":

        chat_history.clear()

        print("\nConversation history cleared!\n")

        continue

    
    if not question:
        continue

    try:

    
        response = rag_chain.invoke({

            "input": question,

            "chat_history": chat_history

        })

       
        answer = response["answer"]

        print("\nAI:", answer)

        
        print("\nSources:")

        seen_sources = set()

        for doc in response["context"]:

            source = doc.metadata.get(
                "source",
                "Unknown"
            )

            if source not in seen_sources:

                print("-", source)

                seen_sources.add(source)

     
        chat_history.append(
            HumanMessage(content=question)
        )

        chat_history.append(
            AIMessage(content=answer)
        )

        print("\n" + "=" * 60 + "\n")

    except Exception as e:

        print("\nSomething went wrong:", e)

        print("Please try asking your question again.\n")
