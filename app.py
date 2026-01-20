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
    """.env와 st.secrets를 모두 지원하는 하이브리드 환경 변수 로더"""
    if key in st.secrets:
        return st.secrets[key]
    return os.getenv(key)

# 페이지 기본 설정
st.set_page_config(
    page_title="Cement Expert AI (Gemini 3)",
    page_icon="🏗️",
    layout="wide"
)

# 2. 로그인 시스템
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login_page():
    st.markdown(
        """
        <style>
        .stTextInput > div > div > input {text-align: center;}
        </style>
        """, unsafe_allow_html=True
    )
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🏗️ Cement Expert Login")
        st.markdown("### 시멘트 공정 관리자 전용")
        st.caption("Powered by Gemini 3 Flash") 
        with st.form("login_form"):
            uid = st.text_input("ID", placeholder="Enter ID")
            upw = st.text_input("Password", type="password", placeholder="Enter Password")
            submit = st.form_submit_button("로그인", use_container_width=True)

            if submit:
                # [보안] 사용자 정보 확인
                if uid == "kilnaid" and upw == "1q2w3e4r":
                    st.session_state.logged_in = True
                    st.success("접속 승인! Gemini 3 시스템에 연결합니다...")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("🚫 ID 또는 비밀번호가 올바르지 않습니다.")

# 3. 메인 애플리케이션 (RAG 시스템)
def main_app():
    # API 클라이언트 연결 (캐싱을 통해 속도 최적화)
    @st.cache_resource
    def init_clients():
        try:
            pc = Pinecone(api_key=get_env("PINECONE_API_KEY"))
            idx = pc.Index(get_env("PINECONE_INDEX_NAME"))
            g_client = genai.Client(api_key=get_env("GEMINI_API_KEY"))
            return idx, g_client
        except Exception as e:
            st.error(f"❌ 서버 연결 실패: {e}")
            return None, None

    index, client = init_clients()
    if not index or not client:
        st.stop()

    # 사이드바
    with st.sidebar:
        st.header("🔧 System Info")
        st.info(f"Connected to: **{get_env('PINECONE_INDEX_NAME')}**")
        st.markdown("---")
        st.markdown("**Model Specs:**")
        st.caption("🧠 LLM: `gemini-3-flash`")
        st.caption("🧮 Embed: `text-embedding-005`")
        st.markdown("---")
        if st.button("로그아웃", use_container_width=True):
            st.session_state.logged_in = False
            st.rerun()

    # 채팅 UI 헤더
    st.title("🏗️ 시멘트 공정 지능형 비서")
    st.caption("🚀 Powered by Gemini 3 Flash & Pinecone Vector Search")

    # 대화 기록 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "반갑습니다, 관리자님. Gemini 3 엔진이 준비되었습니다. 시멘트 공정에 대해 질문해 주세요."}
        ]

    # 대화 기록 출력
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 사용자 입력 처리
    if prompt := st.chat_input("질문을 입력하세요 (예: 킬른 온도가 급격히 오를 때 조치법은?)"):
        # 1. 사용자 질문 표시
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. AI 답변 생성
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            with st.spinner("📚 업로드 문서 정밀 검색 중..."):
                try:
                    # [Step 1] 질문을 벡터로 변환 (최신 모델 text-embedding-005 사용)
                    # 주의: Ingest(업로드)할 때 사용한 모델과 동일해야 검색이 잘 됩니다.
                    # 만약 기존 DB가 004로 되어 있다면, DB를 005로 다시 적재하는 것을 강력 권장합니다.
                    emb_res = client.models.embed_content(
                        model="models/text-embedding-005",
                        contents=prompt
                    )
                    query_vector = emb_res.embeddings[0].values

                    # [Step 2] Pinecone 검색
                    search_res = index.query(
                        vector=query_vector,
                        top_k=5,
                        include_metadata=True
                    )

                    # [Step 3] 검색된 문맥(Context) 조립
                    context_text = ""
                    sources = set()
                    for match in search_res['matches']:
                        meta = match['metadata']
                        context_text += f"\n[출처: {meta.get('source', 'Unknown')} (P.{int(meta.get('page', 0))})]\n{meta.get('text', '')}\n---"
                        sources.add(f"{meta.get('source')} (P.{int(meta.get('page', 0))})")

                    # [Step 4] LLM에게 답변 요청 (Gemini 3 Flash 적용)
                    system_prompt = f"""
                    당신은 20년 경력의 시멘트 공정 관리 전문가입니다.
                    아래 제공된 [기술 문서 내용]을 바탕으로 관리자의 질문에 명확하고 구체적으로 답하세요.
                    
                    - Gemini 3의 뛰어난 추론 능력을 활용하여 복합적인 인과관계를 설명하세요.
                    - 수치나 화학식($CaO$, $C_3S$ 등)이 있다면 정확하게 인용하세요.
                    - 문서에 없는 내용은 "업로드 문서에서 관련 내용을 찾을 수 없습니다"라고 답하고, 웹 검색과 추론을 통해서 보강하여 답변하세요.
                    
                    [기술 문서 내용]:
                    {context_text}
                    """
                    
                    # 스트리밍 답변 생성
                    response = client.models.generate_content(
                        model="gemini-3-flash",  # 최신 모델명
                        contents=[system_prompt, f"질문: {prompt}"],
                        config=types.GenerateContentConfig(temperature=0.1)
                    )
                    
                    full_response = response.text
                    
                    # 출처 표시 추가
                    if sources:
                        full_response += "\n\n**📌 참조 문서:**\n- " + "\n- ".join(sorted(list(sources)))

                    message_placeholder.markdown(full_response)
                
                except Exception as e:
                    # 모델명 에러 발생 시 예비책 안내
                    if "404" in str(e) and "gemini-3-flash" in str(e):
                         st.error("⚠️ 'gemini-3-flash' 모델을 찾을 수 없습니다. API 키 권한을 확인하거나 'gemini-2.0-flash'로 변경해 보세요.")
                    elif "text-embedding-004" in str(e):
                         st.error("⚠️ 임베딩 모델 에러: 005 버전을 사용할 수 없다면, 최신 라이브러리 업데이트가 필요합니다.")
                    else:
                        st.error(f"오류 발생: {e}")
                    full_response = "시스템 오류가 발생했습니다."

            # 대화 기록 저장
            st.session_state.messages.append({"role": "assistant", "content": full_response})

# 메인 실행 로직
if __name__ == "__main__":
    if not st.session_state.logged_in:
        login_page()
    else:
        main_app()