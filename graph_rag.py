import streamlit as st
import openai
from api_config import OPENAI_API_KEY, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, get_neo4j_driver
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.retrievers import Text2CypherRetriever
from neo4j_graphrag.generation import GraphRAG
from neo4j import GraphDatabase, basic_auth

# OpenAI API 설정
openai.api_key = OPENAI_API_KEY

# Neo4j 연결 (session_state에 저장하여 재사용)
if "driver" not in st.session_state:
    st.session_state.driver = get_neo4j_driver()

# 대화 기록 저장
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# LLM 설정
llm = OpenAILLM(model_name="gpt-4o", model_params={"temperature": 0})

# 노드 데이터 타입 반환 함수
def get_node_datatype(value):
    """
        입력된 노드 Value의 데이터 타입을 반환하는 함수
    """
    if isinstance(value, str):
        return "STRING"
    elif isinstance(value, int):
        return "INTEGER"
    elif isinstance(value, float):
        return "FLOAT"
    elif isinstance(value, bool):
        return "BOOLEAN"
    elif isinstance(value, list):
        return f"LIST[{get_node_datatype(value[0])}]" if value else "LIST"
    else:
        return "UNKNOWN"

# Neo4j 스키마 가져오기 함수
def get_schema(uri, user, password):
    driver = GraphDatabase.driver(uri, auth=basic_auth(user, password))
    with driver.session() as session:
        node_query = """
        MATCH (n)
        WITH DISTINCT labels(n) AS node_labels, keys(n) AS property_keys, n
        UNWIND node_labels AS label
        UNWIND property_keys AS key
        RETURN label, key, n[key] AS sample_value
        """
        nodes = session.run(node_query)

        rel_query = """
        MATCH ()-[r]->()
        WITH DISTINCT type(r) AS rel_type, keys(r) AS property_keys, r
        UNWIND property_keys AS key
        RETURN rel_type, key, r[key] AS sample_value
        """
        relationships = session.run(rel_query)

        rel_direction_query = """
        MATCH (a)-[r]->(b)
        RETURN DISTINCT labels(a) AS start_label, type(r) AS rel_type, labels(b) AS end_label
        ORDER BY start_label, rel_type, end_label
        """
        rel_directions = session.run(rel_direction_query)

        schema = {"nodes": {}, "relationships": {}, "relations": []}

        for record in nodes:
            label = record["label"]
            key = record["key"]
            sample_value = record["sample_value"]
            inferred_type = get_node_datatype(sample_value)
            if label not in schema["nodes"]:
                schema["nodes"][label] = {}
            schema["nodes"][label][key] = inferred_type

        for record in relationships:
            rel_type = record["rel_type"]
            key = record["key"]
            sample_value = record["sample_value"]
            inferred_type = get_node_datatype(sample_value)
            if rel_type not in schema["relationships"]:
                schema["relationships"][rel_type] = {}
            schema["relationships"][rel_type][key] = inferred_type

        for record in rel_directions:
            start_label = record["start_label"][0]
            rel_type = record["rel_type"]
            end_label = record["end_label"][0]
            schema["relations"].append(f"(:{start_label})-[:{rel_type}]->(:{end_label})")

        return schema

# 스키마 포맷팅
def format_schema(schema):
    result = []
    result.append("Node properties:")
    for label, properties in schema["nodes"].items():
        props = ", ".join(f"{k}: {v}" for k, v in properties.items())
        result.append(f"{label} {{{props}}}")

    result.append("Relationship properties:")
    for rel_type, properties in schema["relationships"].items():
        props = ", ".join(f"{k}: {v}" for k, v in properties.items())
        result.append(f"{rel_type} {{{props}}}")

    result.append("The relationships:")
    for relation in schema["relations"]:
        result.append(relation)

    return "\n".join(result)

# Neo4j 스키마 가져오기
schema = get_schema(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
neo4j_schema = format_schema(schema)

# LLM INPUT / QUERY 예시 제공
examples = [
    "USER INPUT: '품목기준코드가 200710605인 의약품의 atc 코드가 뭐야?' QUERY: MATCH (d:Drug)-[:CATEGORIZED_AS]->(a:ATC) WHERE d.code = 200710605 RETURN a.code",
    "USER INPUT: '아모라닉정625밀리그램 의약품의 용법용량을 알려줘!' QUERY: MATCH (d:Drug)-[:HAS_DOSAGE]->(dosage:Dosage) WHERE d.name = '아모라닉정625밀리그램' RETURN dosage.description"
]


# Text2CypherRetriever
retriever = Text2CypherRetriever(
    driver=st.session_state.driver,
    llm=llm,  # type: ignore
    neo4j_schema=neo4j_schema,
    examples=examples,
)

# RAG 초기화
rag = GraphRAG(retriever=retriever, llm=llm)

# Streamlit UI
st.title("💬 Neo4j 기반 AI 챗봇")

# 이전 대화 표시
for role, text in st.session_state["chat_history"]:
    with st.chat_message(role):
        st.markdown(text)

# 사용자 입력
user_query = st.chat_input("질문을 입력하세요...")

if user_query:
    # 유저 메시지 추가
    st.session_state["chat_history"].append(("user", user_query))
    with st.chat_message("user"):
        st.markdown(user_query)

    # 스피너 추가
    with st.spinner("답변을 생성 중입니다..."):
        response = rag.search(query_text=user_query)
        response_text = response.answer if response else "데이터를 찾을 수 없습니다."
    
    # 결과를 대화 히스토리에 추가
    st.session_state["chat_history"].append(("assistant", response_text))

    with st.chat_message("assistant"):
        st.markdown(response_text)