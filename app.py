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
        # Pinecone v6+ 및 Gemini 3 SDK 초기화
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
        st.markdown("**분석 가이드:**")
        st.caption("- 하이브리드 지식 결합 (내부 문서 + 실시간 웹)")
        st.caption("- 공정 인과관계 및 열역학적 추론")
        st.markdown("---")
        if st.button("로그아웃", use_container_width=True):
            st.session_state.logged_in = False
            st.rerun()

    st.title("🏗️ 시멘트 생산·품질 지능형 비서")
    st.caption("🚀 Gemini 3 Flash & Multi-Document RAG Insight")

    # 대화 기록 관리
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "반갑습니다, 관리자님. 시멘트 생산 및 품질 전반에 대해 무엇이든 말씀해 주세요. 전문 문서를 바탕으로 심도 있게 분석해 드리겠습니다."}
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 질문 입력
    if prompt := st.chat_input("공정 이슈나 기술적인 질문을 입력하세요..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("전문 지식 심층 분석 중..."):
                try:
                    # [Step 1] 검색 범위 확대 (top_k=15)
                    # 42개 이상의 문서를 충분히 검토하기 위해 검색 결과를 늘립니다.
                    search_res = index.query(
                        data=prompt, 
                        top_k=15, 
                        include_metadata=True
                    )

                    context_text = ""
                    sources = set()
                    for match in search_res['matches']:
                        meta = match['metadata']
                        context_text += f"\n[출처: {meta.get('source')} (P.{meta.get('page')})]\n{meta.get('text', '')}\n---"
                        sources.add(f"{meta.get('source')} (P.{int(meta.get('page', 0))})")

                    # [Step 2] 시스템 프롬프트 고도화 (형식 파괴 및 깊이 강조)
                    system_prompt = f"""
                    당신은 30년 경력의 세계 최고 시멘트 공정 및 품질 관리 기술 고문입니다.
                    단순히 정보를 나열하는 챗봇이 아니라, 관리자의 고민에 대해 '원인-이론-해결책'의 유기적인 인과관계를 꿰뚫는 깊이 있는 통찰을 제공하세요.

                    [응답 지침]
                    1. **심층적 인과관계 분석**: 표면적인 현상보다 그 이면에 숨겨진 열역학적, 화학적 메커니즘을 설명하세요. Gemini 3의 뛰어난 추론 능력을 활용하여 복합적인 변수들 간의 상관관계를 파악하세요.
                    2. **풍부한 지식 활용**: 제공된 [기술 문서 내용]을 꼼꼼히 검토하여 수치, 화학식, 설비 사양 등을 구체적으로 인용하며 신뢰도를 높이세요.
                    3. **자유로운 서술 형식**: 억지로 포맷을 맞추려 하지 말고, 전문가가 대화하듯 논리적이고 유려하게 답변하세요. 필요하다면 항목별 요약을 곁들이되 전체적인 설명의 깊이를 우선시하세요.
                    4. **하이브리드 지식 결합**: 문서에 없는 내용은 구글 검색을 통해 확보한 최신 기술 트렌드와 당신의 공학적 추론을 결합하여 답변을 보강하세요.
                    5. **전문가적 제언**: 관리자가 미처 생각하지 못한 공정상의 유연성(Buffer), 설비 안정성, 원료 균일성 등의 관점에서도 조언을 아끼지 마세요.

                    [기술 문서 내용]:
                    {context_text}
                    """

                    # [Step 3] Gemini 3 호출 (웹 검색 도구 포함)
                    google_search_tool = types.Tool(google_search=types.GoogleSearch())
                    
                    response = client.models.generate_content(
                        model="gemini-3-flash",
                        contents=f"{system_prompt}\n\n질문: {prompt}",
                        config=types.GenerateContentConfig(
                            tools=[google_search_tool],
                            temperature=0.2 # 약간의 창의성과 유연성을 위해 온도를 소폭 조절
                        )
                    )
                    
                    full_response = response.text
                    if sources:
                        full_response += "\n\n**📌 분석 참조 기술 문서:**\n- " + "\n- ".join(sorted(list(sources)))

                    st.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})

                except Exception as e:
                    st.error(f"⚠️ 처리 중 오류 발생: {e}")

if __name__ == "__main__":
    if not st.session_state.logged_in:
        login_page()
    else:
        main_app()
