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

# 2. 로그인 시스템 (보안 정책 적용)
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
                    st.success("인증 성공! 공정 비서를 가동합니다.")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("❌ 잘못된 자격 증명입니다.")

# 3. 메인 채팅 애플리케이션
def main_app():
    @st.cache_resource
    def init_clients():
        # Pinecone 및 Gemini 3 SDK 초기화
        pc = Pinecone(api_key=get_env("PINECONE_API_KEY"))
        idx = pc.Index(get_env("PINECONE_INDEX_NAME"))
        g_client = genai.Client(api_key=get_env("GEMINI_API_KEY"))
        return idx, g_client

    index, client = init_clients()

    with st.sidebar:
        st.header("🔧 시스템 상태")
        st.success("데이터베이스 연결됨")
        st.info("엔진: Gemini 3 Flash Preview") #
        st.markdown("---")
        if st.button("로그아웃", use_container_width=True):
            st.session_state.logged_in = False
            st.rerun()

    st.title("🏗️ 시멘트 생산·품질 지능형 비서")
    st.caption("🚀 Gemini 3 Flash & Deep-Dive RAG Insight (Error Fixed)")

    # 대화 기록 관리
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "반갑습니다, 관리자님. 30년 경력의 시멘트 기술 고문으로서 공정 전반에 대한 심도 있는 분석을 시작합니다."}
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 질문 입력 및 처리
    if prompt := st.chat_input("분석이 필요한 공정 이슈를 입력하세요..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("전문 기술 문서 정밀 분석 및 추론 중..."):
                try:
                    # [Step 1] 수동 임베딩 생성 (400 에러 해결의 핵심)
                    # 기존 학습 데이터와 동일한 004 모델을 사용하여 768 벡터 생성
                    emb_res = client.models.embed_content(
                        model="models/text-embedding-004",
                        contents=prompt
                    )
                    query_vector = emb_res.embeddings[0].values

                    # [Step 2] Pinecone 벡터 검색 (top_k=15 확대)
                    search_res = index.query(
                        vector=query_vector, # 'data' 대신 직접 'vector' 전달
                        top_k=15, 
                        include_metadata=True
                    )

                    context_text = ""
                    sources = set()
                    for match in search_res['matches']:
                        meta = match['metadata']
                        context_text += f"\n[출처: {meta.get('source')} (P.{meta.get('page')})]\n{meta.get('text', '')}\n---"
                        sources.add(f"{meta.get('source')} (P.{int(meta.get('page', 0))})")

                    # [Step 3] 30년 경력 기술 고문 페르소나 적용 (형식적 제약 제거)
                    system_prompt = f"""
                    당신은 30년 경력의 시멘트 공정 및 품질 관리 분야 세계 최고 기술 고문입니다. 
                    단순한 요약이 아니라, 관리자의 질문에 대해 현상의 본질을 꿰뚫는 '원인-이론-대책'의 유기적인 인과관계를 설명하세요.

                    [가이드라인]
                    1. **심층 추론**: 킬른 내부의 열역학적 변화, 화학적 상 평형($CaO$, $C_3S$ 등), 설비 물리적 거동 간의 복합적 상관관계를 분석하세요.
                    2. **맥락 활용**: 제공된 [기술 문서 내용]에 포함된 구체적인 수치와 도표 데이터를 적극적으로 인용하여 답변의 전문성을 높이세요.
                    3. **유연한 서술**: 억지로 포맷에 맞추기보다, 전문가가 대화하듯 논리적이고 유려하게 답변하세요.
                    4. **하이브리드 지식**: 문서에 없는 내용은 구글 검색 정보와 당신의 공학적 추론을 결합하여 'Deep Insight'를 제공하세요.

                    [기술 문서 내용]:
                    {context_text}
                    """

                    # [Step 4] Gemini 3 호출 (웹 검색 도구 포함)
                    google_search_tool = types.Tool(google_search=types.GoogleSearch())
                    
                    response = client.models.generate_content(
                        model="gemini-3-flash-preview", # 안정적인 최신 모델명
                        contents=f"{system_prompt}\n\n분석 요청: {prompt}",
                        config=types.GenerateContentConfig(
                            tools=[google_search_tool],
                            temperature=0.3
                        )
                    )
                    
                    full_response = response.text
                    if sources:
                        full_response += "\n\n**📌 분석 참조 기술 문서:**\n- " + "\n- ".join(sorted(list(sources)))

                    st.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})

                except Exception as e:
                    st.error(f"⚠️ 시스템 오류 발생: {e}")
                    st.info("💡 Tip: Pinecone 인덱스 치수가 768이 맞는지 다시 한번 확인해 주세요.")

if __name__ == "__main__":
    if not st.session_state.logged_in:
        login_page()
    else:
        main_app()
