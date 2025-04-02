import os
from dotenv import load_dotenv
from neo4j import GraphDatabase, basic_auth

# .env 파일 로드 진행행
load_dotenv()

# 환경 변수에서 API 키 불러오기
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Neo4j 데이터베이스 연결 설정
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

def get_neo4j_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=basic_auth(NEO4J_USER, NEO4J_PASSWORD))
