from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory

from config import answer_examples


# 1. 문서들을 하나의 문자열로 합쳐주는 함수 (create_stuff_documents_chain 대체)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

store = {}



def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    세션 ID에 해당하는 대화 이력을 반환합니다.
    ChatMessageHistory 대신 안정적인 InMemoryChatMessageHistory를 사용합니다.
    """
    if session_id not in store:
        # 이 부분이 핵심 대체 코드입니다.
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

def get_retriever():
    #Embedding : 텍스트를 컴퓨터가 이해할 수 있는 숫자리스트(벡터값)으로 변환하는 변환기
    embedding = OpenAIEmbeddings(model='text-embedding-3-large')
    index_name = 'real-estate-index'
    #
    database = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding)
    retriever = database.as_retriever(search_kwargs={'k': 4})
    return retriever

def get_history_retriever():
    llm = get_llm()
    retriever = get_retriever()
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    history_aware_retriever = (
        contextualize_q_prompt 
        | llm 
        | StrOutputParser() 
        | retriever
    )
    return history_aware_retriever


def get_llm(model='gpt-4o'):
    llm = ChatOpenAI(model=model)
    return llm


def get_dictionary_chain():
    dictionary = ["사람을 나타내는 표현 -> 거주자"]
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(f"""
        사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
        만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
        그런 경우에는 질문만 리턴해주세요
        사전: {dictionary}
        
        질문: {{question}}
    """)

    dictionary_chain = prompt | llm | StrOutputParser()
    
    return dictionary_chain


def get_rag_chain():
    llm = get_llm()
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=answer_examples,
    )
    system_prompt = (
        "당신은 부동산거래신고법 전문가입니다. 사용자의 부동산거래신고법 관한 질문에 답변해주세요"
        "아래에 제공된 문서를 활용해서 답변해주시고"
        "답변을 알 수 없다면 모른다고 답변해주세요"
        "\n\n"
        "{context}"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt,
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = get_history_retriever()
    
    rag_chain = (
        RunnableParallel({
            "context": history_aware_retriever | format_docs,
            "input": lambda x: x["input"],
            "chat_history": lambda x: x["chat_history"]
        })
        | qa_prompt 
        | llm 
        | StrOutputParser()
    )
    
    # 4. 메모리 연결 (RunnableWithMessageHistory 설정)
    # output_messages_key는 rag_chain이 문자열을 반환하므로 None으로 설정하거나 생략합니다.
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history"
    )

    return conversational_rag_chain


def get_ai_response(user_message):
    dictionary_chain = get_dictionary_chain()
    rag_chain = get_rag_chain()
    real_estate_chain = {"input": dictionary_chain} | rag_chain
    ai_response = real_estate_chain.stream(
        {
            "question": user_message
        },
        config={
            "configurable": {"session_id": "abc123"}
        },
    )

    return ai_response