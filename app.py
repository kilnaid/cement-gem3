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
    """배포(Secrets)와 로컬(.env) 환경 변수 통합 관리"""
    if key in st.secrets:
        return st.secrets[key]
    return os.getenv(key)

# 페이지 레이아웃 및 제목 설정
st.set_page_config(
    page_title="Cement Process Insight (Gemini 3)",
    page_icon="🏗️",
    layout="wide"
)

# 2. 로그인 시스템 (보안 정책 적용)
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login_page():
    st.markdown("<h2 style='text-align: center;'>🏗️ Cement Process Expert Login</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form"):
            uid = st.text_input("ID", placeholder="Manager ID")
            upw = st.text_input("Password", type="password", placeholder="Manager Password")
            if st.form_submit_button("시스템 접속", use_container_width=True):
                # 사용자 요약 정보 기반 인증
                if uid == "kilnaid" and upw == "1q2w3e4r":
                    st.session_state.logged_in = True
                    st.success("인증 성공! 전문 보고서 시스템에 연결되었습니다.")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("❌ 자격 증명이 올바르지 않습니다.")

# 3. 메인 채팅 애플리케이션
def main_app():
    @st.cache_resource
    def init_clients():
        # Pinecone 및 Gemini 3 전용 클라이언트 초기화
        pc = Pinecone(api_key=get_env("PINECONE_API_KEY"))
        idx = pc.Index(get_env("PINECONE_INDEX_NAME"))
        g_client = genai.Client(api_key=get_env("GEMINI_API_KEY"))
        return idx, g_client

    index, client = init_clients()

    # 사이드바 설정
    with st.sidebar:
        st.header("🔧 시스템 상태")
        st.success("데이터베이스 연결됨 (v.004)")
        st.info("엔진: Gemini 3 Flash Preview")
        st.markdown("---")
        st.markdown("**보고서 분석 항목:**")
        st.markdown("---")
        if st.button("로그아웃", use_container_width=True):
            st.session_state.logged_in = False
            st.rerun()

    st.title("🏗️ 시멘트 공정 지능형 분석 비서")
    st.caption("🚀 시멘트 AI가 분석하는 3단계 기술 보고서 시스템")

    # 대화 기록 관리
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "관리자님, 분석이 필요한 이슈를 말씀해 주세요."}
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 질문 입력 및 처리
    if prompt := st.chat_input("이슈를 입력하세요 (예: 킬른 화염 안정성 저하 시 조치법)"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("전문 문서를 분석하고 웹 검색을 통해 보고서를 작성 중입니다..."):
                try:
                    # [Step 1] 적재된 데이터 호환 임베딩 (004 모델 사용)
                    emb_res = client.models.embed_content(
                        model="models/text-embedding-004",
                        contents=prompt
                    )
                    query_vector = emb_res.embeddings[0].values

                    # [Step 2] Pinecone 벡터 검색
                    search_res = index.query(
                        vector=query_vector,
                        top_k=5,
                        include_metadata=True
                    )

                    context_text = ""
                    sources = set()
                    for match in search_res['matches']:
                        meta = match['metadata']
                        context_text += f"\n[출처: {meta.get('source')} (P.{meta.get('page')})]\n{meta.get('text', '')}\n---"
                        sources.add(f"{meta.get('source')} (P.{int(meta.get('page', 0))})")

                    # [Step 3] 고도화된 보고서형 시스템 프롬프트
                    # 수치 및 화학식 표현을 위한 LaTeX 가이드 포함
                    system_prompt = f"""
                    당신은 30년 경력의 시멘트 공정 관리 전문가입니다.
                    관리자의 질문에 대해 아래 3단계 구조의 전문 기술 보고서 형식으로 답변하세요.

                    ### 1. 현상 분석 및 문제 원인 파악 (Problem & Causes)
                    - 제공된 [기술 문서 내용]과 실시간 웹 검색 정보를 종합하여 현상을 정의하세요.
                    - 근본 원인(Root Causes)을 논리적으로 나열하세요.
                    - 문서에 없는 내용은 "매뉴얼 외 최신 기술 동향 검색 결과"임을 명시하고 설명하세요.

                    ### 2. 기술적 배경 및 관련 이론 (Technical Background)
                    - 시멘트 화학($CaO$, $C_3S$, $C_2S$, $C_3A$, $C_4AF$) 및 열역학 원리를 설명하세요.
                    - Gemini 3의 추론 능력을 사용하여 복합적인 인과관계(예: Burner 확장과 소성 온도 상관관계)를 분석하세요.

                    ### 3. 공정 현장 적용 방안 (Action Plan)
                    - 관리자가 현장에서 즉시 실행할 수 있는 실무적 대책을 제안하세요.
                    - 예상되는 개선 결과 및 주의사항을 명확히 제시하세요.

                    [기술 문서 내용]:
                    {context_text}
                    """

                    # [Step 4] Gemini 3 호출 (웹 검색 도구 통합)
                    google_search_tool = types.Tool(google_search=types.GoogleSearch())
                    
                    response = client.models.generate_content(
                        model="gemini-3-flash-preview", 
                        contents=f"{system_prompt}\n\n분석 요청 사항: {prompt}",
                        config=types.GenerateContentConfig(
                            tools=[google_search_tool],
                            temperature=0.1
                        )
                    )
                    
                    full_response = response.text
                    if sources:
                        full_response += "\n\n**📌 참조 기술 문서:**\n- " + "\n- ".join(sorted(list(sources)))

                    st.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})

                except Exception as e:
                    st.error(f"⚠️ 보고서 작성 중 오류 발생: {e}")

if __name__ == "__main__":
    if not st.session_state.logged_in:
        login_page()
    else:
        main_app()
