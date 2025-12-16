import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
# import requests
# import numpy as np
# from langchain.document_loaders import TextLoader, PyMuPDFLoader, UnstructuredPowerPointLoader, UnstructuredWordDocumentLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS, Chroma
# from langchain.tools import Tool



# 사용할 모델 선택 
model_name = "./DeepSeek-R1-Distill-Qwen-14B"

def load_tokenizer():
    return AutoTokenizer.from_pretrained(model_name)

def load_model():
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )



# 모델 테스트 함수
def generate_answer(prompt):
    # 사용자 입력을 메시지 리스트에 추가
    st.session_state["messages"].append({"role": "user", "content": user_input})

    # 최근 대화 2개를 모아 프롬프트 구성
    chat_history = st.session_state["messages"][-2:]
    prompt = ""
    for msg in chat_history:
        role = msg["role"]
        content = msg["content"]
        prompt += f"{'User' if role == 'user' else 'Assistant'}: {content}\n"
    prompt += "Assistant:"

    # 모델 및 토크나이저 GPU에 계속 유지
    model = st.session_state["model"]
    tokenizer = st.session_state["tokenizer"]

    # 챗봇 응답
    with torch.no_grad():
        inputs = st.session_state["tokenizer"](prompt, return_tensors="pt").to("cuda")
        outputs = st.session_state["model"].generate(**inputs, max_new_tokens=500)
        response = st.session_state["tokenizer"].decode(outputs[0], skip_special_tokens=True)

    st.session_state["messages"].append({"role": "assistant", "content": response})
    
    return response
if "model" not in st.session_state:
    st.session_state["model"] = load_model().to("cuda")
    st.session_state["tokenizer"] = load_tokenizer()

# Streamlit UI 설정
st.title("🤖 DeepSeek R1 챗봇")
st.write("DeepSeek AI 모델과 대화하세요!")

# 채팅 기록 저장
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 이전 메시지 표시
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# 문서 업로드

# 벡터 DB 저장 - LAG

# FAST API 적용



# streamlit UI 파트트
# 사용자 입력
user_input = st.chat_input("메시지를 입력하세요...")
if user_input:
    # 사용자 메시지 추가
    st.session_state["messages"].append({"role": "user", "content": user_input})
    
    # UI에 사용자 메시지 표시
    with st.chat_message("user"):
        st.markdown(user_input)

    # DeepSeek 모델 직접 호출
    try:
        bot_reply = generate_answer(user_input)
    except Exception as e:
        bot_reply = f"오류 발생: {str(e)}"

    # 챗봇 응답 저장 및 표시
    st.session_state["messages"].append({"role": "assistant", "content": bot_reply})

    with st.chat_message("assistant"):
        st.markdown(bot_reply)

