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

# 페이지 레이아웃 설정
st.set_page_config(
    page_title="Cement Expert AI (Gemini 3)",
    page_icon="🏗️",
    layout="wide"
)

# 2. 로그인 시스템 (보안 정책 적용)
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login_page():
    st.markdown("<h2 style='text-align: center;'>🏗️ Cement Expert AI Login</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form"):
            uid = st.text_input("ID", placeholder="Enter ID")
            upw = st.text_input("Password", type="password", placeholder="Enter Password")
            if st.form_submit_button("시스템 접속", use_container_width=True):
                # 사용자 요약 기반 인증 데이터
                if uid == "kilnaid" and upw == "1q2w3e4r":
                    st.session_state.logged_in = True
                    st.success("인증 성공! 공정 지휘 본부에 연결되었습니다.")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("❌ 자격 증명이 올바르지 않습니다.")

# 3. 메인 채팅 애플리케이션
def main_app():
    @st.cache_resource
    def init_clients():
        # Pinecone v6+ 및 Gemini 3 전용 클라이언트 초기화
        pc = Pinecone(api_key=get_env("PINECONE_API_KEY"))
        idx = pc.Index(get_env("PINECONE_INDEX_NAME"))
        g_client = genai.Client(api_key=get_env("GEMINI_API_KEY"))
        return idx, g_client

    index, client = init_clients()

    # 사이드바 설정
    with st.sidebar:
        st.header("🔧 시스템 상태")
        st.success("데이터베이스 연결됨")
        st.info("엔진: Gemini 3 Flash")
        st.markdown("---")
        st.markdown("**관리 중인 공정 정보:**")
        st.caption("- 킬른 화염 안정성 분석")
        st.caption("- 클린커 품질($f-CaO$) 모니터링")
        st.markdown("---")
        if st.button("로그아웃", use_container_width=True):
            st.session_state.logged_in = False
            st.rerun()

    st.title("🏗️ 시멘트 공정 지능형 비서")
    st.caption("🚀 Gemini 3 Flash & Pinecone RAG System (768 Dimension Optimized)")

    # 대화 기록 관리
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "반갑습니다, 관리자님. 20년 경력의 시멘트 공정 지식을 바탕으로 지원하겠습니다. 무엇을 도와드릴까요?"}
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 질문 입력 및 처리
    if prompt := st.chat_input("공정 이상 징후나 기술 매뉴얼에 대해 질문하세요..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("전문 기술 문서 분석 및 웹 검색 병행 중..."):
                try:
                    # [Step 1] 수동 임베딩 (768 Dimension 매칭 - 400 에러 해결 핵심)
                    # Index가 768이므로 text-embedding-004를 명시적으로 사용합니다.
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

                    # [Step 3] 시스템 프롬프트 구성 (사용자 요청사항 반영)
                    system_prompt = f"""
                    당신은 20년 경력의 시멘트 공정 관리 전문가입니다.
                    아래 제공된 [기술 문서 내용]을 바탕으로 관리자의 질문에 명확하고 구체적으로 답하세요.
                    
                    - Gemini 3의 뛰어난 추론 능력을 활용하여 복합적인 인과관계를 설명하세요. (예: 버너 열팽창과 화염 안정성, $f-CaO$와 소성 온도 관계 등)
                    - 수치나 화학식($CaO$, $C_3S$ 등)이 있다면 정확하게 인용하세요.
                    - 문서에 없는 내용은 "업로드 문서에서 관련 내용을 찾을 수 없습니다"라고 답하고, 웹 검색과 추론을 통해서 보강하여 답변하세요.

                    [기술 문서 내용]:
                    {context_text}
                    """

                    # [Step 4] Gemini 3 호출 (웹 검색 도구 통합)
                    google_search_tool = types.Tool(google_search=types.GoogleSearch())
                    
                    response = client.models.generate_content(
                        model="gemini-3-flash-preview",
                        contents=f"{system_prompt}\n\n질문: {prompt}",
                        config=types.GenerateContentConfig(
                            tools=[google_search_tool],
                            temperature=0.1
                        )
                    )
                    
                    full_response = response.text
                    if sources:
                        full_response += "\n\n**📌 참조 문서:**\n- " + "\n- ".join(sorted(list(sources)))

                    st.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})

                except Exception as e:
                    st.error(f"⚠️ 시스템 오류 발생: {e}")
                    st.info("💡 팁: Pinecone 인덱스의 치수(Dimension)가 768이 맞는지 확인해 주세요.")

if __name__ == "__main__":
    if not st.session_state.logged_in:
        login_page()
    else:
        main_app()


