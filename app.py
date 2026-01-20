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
    """배포 환경(Secrets)과 로컬(.env) 환경 변수 통합 로드"""
    if key in st.secrets:
        return st.secrets[key]
    return os.getenv(key)

# 페이지 레이아웃 설정
st.set_page_config(
    page_title="Cement Expert AI (Gemini 3)",
    page_icon="🏗️",
    layout="wide"
)

# 2. 로그인 시스템
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
                # 사용자 고유 계정 정보 활용
                if uid == "kilnaid" and upw == "1q2w3e4r":
                    st.session_state.logged_in = True
                    st.success("인증 성공! 공정 비서를 가동합니다.")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("❌ 잘못된 자격 증명입니다.")

# 3. 메인 채팅 애플리케이션
def main_app():
    @st.cache_resource
    def init_clients():
        # 2026년형 Pinecone v6+ 및 Gemini 3 SDK 초기화
        pc = Pinecone(api_key=get_env("PINECONE_API_KEY"))
        idx = pc.Index(get_env("PINECONE_INDEX_NAME"))
        g_client = genai.Client(api_key=get_env("GEMINI_API_KEY"))
        return idx, g_client

    index, client = init_clients()

    with st.sidebar:
        st.header("🔧 시스템 상태")
        st.success("데이터베이스 연결됨")
        st.info("엔진: Gemini 3 Flash")
        st.markdown("---")
        if st.button("로그아웃", use_container_width=True):
            st.session_state.logged_in = False
            st.rerun()

    st.title("🏗️ 시멘트 공정 지능형 비서")
    st.caption("🚀 Gemini 3 Flash & Pinecone Integrated RAG System")

    # 대화 기록 관리
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "반갑습니다, 관리자님. 20년 경력의 시멘트 공정 지식을 바탕으로 지원하겠습니다."}
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 질문 입력
    if prompt := st.chat_input("질문을 입력하세요..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("전문 지식 분석 및 웹 검색 병행 중..."):
                try:
                    # [Step 1] Pinecone 통합 임베딩 검색 (Inference API)
                    search_res = index.query(
                        data=prompt, 
                        top_k=5, 
                        include_metadata=True
                    )

                    context_text = ""
                    sources = set()
                    for match in search_res['matches']:
                        meta = match['metadata']
                        context_text += f"\n[출처: {meta.get('source')} (P.{meta.get('page')})]\n{meta.get('text', '')}\n---"
                        sources.add(f"{meta.get('source')} (P.{int(meta.get('page', 0))})")

                    # [Step 2] 시스템 프롬프트 구성 (사용자 요청 반영)
                    system_prompt = f"""
                    당신은 20년 경력의 시멘트 공정 관리 전문가입니다.
                    아래 제공된 [기술 문서 내용]을 바탕으로 관리자의 질문에 명확하고 구체적으로 답하세요.
                    
                    - Gemini 3의 뛰어난 추론 능력을 활용하여 복합적인 인과관계를 설명하세요.
                    - 수치나 화학식($CaO$, $C_3S$ 등)이 있다면 정확하게 인용하세요.
                    - 문서에 없는 내용은 "업로드 문서에서 관련 내용을 찾을 수 없습니다"라고 답하고, 웹 검색과 추론을 통해서 보강하여 답변하세요.

                    [기술 문서 내용]:
                    {context_text}
                    """

                    # [Step 3] Gemini 3 호출 (웹 검색 도구 포함)
                    # 2026년형 SDK의 구글 검색 도구 설정
                    google_search_tool = types.Tool(google_search=types.GoogleSearch())
                    
                    response = client.models.generate_content(
                        model="gemini-3-flash",
                        contents=f"{system_prompt}\n\n질문: {prompt}",
                        config=types.GenerateContentConfig(
                            tools=[google_search_tool],
                            temperature=0.1
                        )
                    )
                    
                    full_response = response.text
                    if sources:
                        full_response += "\n\n**📌 문서 참조:**\n- " + "\n- ".join(sorted(list(sources)))

                    st.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})

                except Exception as e:
                    st.error(f"⚠️ 처리 중 오류 발생: {e}")

if __name__ == "__main__":
    if not st.session_state.logged_in:
        login_page()
    else:
        main_app()
