import streamlit as st
import os
import time
from pinecone import Pinecone
from google import genai
from google.genai import types
from dotenv import load_dotenv

# 1. 환경 설정 및 초기화
load_dotenv()

def get_env(key):
    if key in st.secrets:
        return st.secrets[key]
    return os.getenv(key)

st.set_page_config(page_title="Cement Expert AI (Full Memory)", page_icon="🏗️", layout="wide")

# 2. 로그인 시스템 (기존 유지)
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login_page():
    st.markdown("<h2 style='text-align: center;'>🏗️ Cement Expert AI Login</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form"):
            uid = st.text_input("ID", placeholder="Manager ID")
            upw = st.text_input("Password", type="password", placeholder="Manager Password")
            if st.form_submit_button("시스템 접속", use_container_width=True):
                if uid == "kilnaid" and upw == "1q2w3e4r": #
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error("❌ 자격 증명이 올바르지 않습니다.")

# 3. 메인 채팅 애플리케이션
def main_app():
    @st.cache_resource
    def init_clients():
        pc = Pinecone(api_key=get_env("PINECONE_API_KEY"))
        idx = pc.Index(get_env("PINECONE_INDEX_NAME"))
        g_client = genai.Client(api_key=get_env("GEMINI_API_KEY"))
        return idx, g_client

    index, client = init_clients()

    with st.sidebar:
        st.header("🔧 시스템 상태")
        st.success("데이터베이스 연결됨 (RAG)")
        st.info("지능형 메모리 활성화 (Full History)")
        st.markdown("---")
        if st.button("로그아웃", use_container_width=True):
            st.session_state.logged_in = False
            st.rerun()

    st.title("🏗️ 시멘트 생산·품질 기술 고문")
    st.caption("🚀 Gemini 3 Flash & Multi-Turn Conversation Memory")

    # [중요] 대화 기록 관리 및 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = [] # 빈 리스트로 시작 (첫 메시지는 루프 밖에서 처리)

    # 대화 기록 출력 (UI)
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 질문 입력
    if prompt := st.chat_input("공정 이슈나 지난 대화에 이어 질문하세요..."):
        # 1. 사용자 질문 UI 표시 및 저장
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("과거 대화와 전문 문서를 종합 분석 중..."):
                try:
                    # [Step 1] 수동 임베딩 및 검색 (400 에러 방지)
                    emb_res = client.models.embed_content(model="models/text-embedding-004", contents=prompt)
                    search_res = index.query(vector=emb_res.embeddings[0].values, top_k=15, include_metadata=True)

                    context_text = ""
                    sources = set()
                    for match in search_res['matches']:
                        meta = match['metadata']
                        context_text += f"\n[출처: {meta.get('source')} (P.{meta.get('page')})]\n{meta.get('text', '')}\n---"
                        sources.add(f"{meta.get('source')} (P.{int(meta.get('page', 0))})")

                    # [Step 2] 시스템 지침 구성
                    system_instruction = f"""
                    당신은 30년 경력의 시멘트 기술 고문입니다. 
                    관리자와의 대화 흐름을 완벽히 파악하여, 이전 질문에서 다룬 맥락을 유지하며 답변하세요.

                    - 이번 질문에 대한 전문 문서 근거:
                    {context_text}

                    [지침]
                    1. 과거 대화에 언급된 설비나 특정 수치를 기억하고 이를 바탕으로 추론하세요.
                    2. 형식에 얽매이지 말고, 전문가가 기술 보고서를 작성하듯 심도 있고 자세하고, 길게 서술하세요.
                    3. 문서에 없는 내용은 웹 검색과 당신의 공학적 지식을 결합하여 통찰을 제공하세요.
                    4. **맥락 활용**: 제공된 [기술 문서 내용]에 포함된 구체적인 수치와 도표 데이터를 적극적으로 인용하여 답변의 전문성을 높이세요.
                    5. **유연한 서술**: 억지로 포맷에 맞추기보다, 전문가가 대화하듯 논리적이고 유려하게 답변하세요.
                    6. **하이브리드 지식**: 문서에 없는 내용은 구글 검색 정보와 당신의 공학적 추론을 결합하여 'Deep Insight'를 제공하세요.
                    """

                    # [Step 3] 대화 기록(History) 재구성 (Gemini API 형식에 맞춤)
                    # 과거 메시지들을 Gemini가 이해할 수 있는 형태의 'contents' 리스트로 변환합니다.
                    chat_history = []
                    for m in st.session_state.messages[:-1]: # 마지막 질문 제외한 과거 기록
                        role = "user" if m["role"] == "user" else "model"
                        chat_history.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))

                    # [Step 4] 답변 생성 (전체 히스토리 + 시스템 지침 + 현재 질문)
                    google_search_tool = types.Tool(google_search=types.GoogleSearch())
                    
                    response = client.models.generate_content(
                        model="gemini-3-flash-preview",
                        contents=chat_history + [
                            types.Content(role="user", parts=[types.Part(text=f"{system_instruction}\n\n최종 질문: {prompt}")])
                        ],
                        config=types.GenerateContentConfig(tools=[google_search_tool], temperature=0.3)
                    )
                    
                    full_response = response.text
                    if sources:
                        full_response += "\n\n**📌 참조 문서:**\n- " + "\n- ".join(sorted(list(sources)))

                    st.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})

                except Exception as e:
                    st.error(f"⚠️ 처리 중 오류 발생: {e}")

if __name__ == "__main__":
    if not st.session_state.logged_in: login_page()
    else: main_app()
