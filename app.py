import streamlit as st
import os
import time
import io
from pinecone import Pinecone
from google import genai
from google.genai import types
from dotenv import load_dotenv
import pandas as pd
from PIL import Image

# 1. 환경 설정 및 초기화
load_dotenv()

def get_env(key):
    """배포 환경(Secrets)과 로컬(.env) 환경 변수 통합 로드"""
    if key in st.secrets:
        return st.secrets[key]
    return os.getenv(key)


def build_uploaded_file_context(uploaded_file):
    """업로드 파일 요약(표) 또는 이미지 파트를 생성해 추론 컨텍스트로 반환."""
    if uploaded_file is None:
        return "", None

    file_name = uploaded_file.name
    file_bytes = uploaded_file.getvalue()
    ext = os.path.splitext(file_name)[1].lower()

    if ext in [".xlsx", ".xls", ".csv"]:
        try:
            if ext == ".csv":
                df = pd.read_csv(io.BytesIO(file_bytes))
                sheet_name = "csv"
            else:
                xls = pd.ExcelFile(io.BytesIO(file_bytes))
                sheet_name = xls.sheet_names[0]
                df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name)

            head_text = df.head(5).to_csv(index=False)
            col_preview = ", ".join(map(str, df.columns[:40]))
            context = (
                f"Uploaded file: {file_name}\n"
                f"Type: tabular\n"
                f"Sheet: {sheet_name}\n"
                f"Rows: {len(df)}, Columns: {len(df.columns)}\n"
                f"Columns preview: {col_preview}\n"
                f"Top 5 rows (CSV):\n{head_text}"
            )
            return context, None
        except Exception as e:
            return f"Uploaded tabular file parse failed: {file_name}, error: {e}", None

    if ext in [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"]:
        try:
            img = Image.open(io.BytesIO(file_bytes))
            mime = uploaded_file.type or "image/png"
            image_part = types.Part.from_bytes(data=file_bytes, mime_type=mime)
            context = (
                f"Uploaded file: {file_name}\n"
                f"Type: image\n"
                f"Image size: {img.width}x{img.height}, mode: {img.mode}\n"
                "Use this image as additional evidence for analysis."
            )
            return context, image_part
        except Exception as e:
            return f"Uploaded image parse failed: {file_name}, error: {e}", None

    return f"Unsupported uploaded file type: {file_name}", None

# 모델 및 인덱스 규격 설정
EMBED_MODEL = "models/gemini-embedding-001"
CHAT_MODEL = "models/gemini-3-flash-preview"
TARGET_DIMENSION = 768  # 관리자님의 Pinecone 인덱스 차원 규격

st.set_page_config(page_title="Cement Expert AI (Deep Insight)", page_icon="🏗️", layout="wide")

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
                if uid == "sampyo" and upw == "1q2w3e4r":
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error("❌ 자격 증명이 올바르지 않습니다.")

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
        st.success("데이터베이스 연결됨 (RAG)")
        st.info(f"임베딩: {EMBED_MODEL} (768d)")
        uploaded_file = st.file_uploader(
            "Upload Excel/Image for current analysis",
            type=["xlsx", "xls", "csv", "png", "jpg", "jpeg", "bmp", "gif", "webp"],
            accept_multiple_files=False,
            help="The uploaded file is included as context in AI reasoning for your question.",
        )
        if uploaded_file is not None:
            st.success(f"Uploaded: {uploaded_file.name}")
        st.markdown("---")
        if st.button("로그아웃", use_container_width=True):
            st.session_state.logged_in = False
            st.rerun()

    st.title("🏗️ 시멘트 생산·품질 기술 고문")
    st.caption(f"🚀 {CHAT_MODEL} & Deep-Dive RAG Insight")

    # 대화 기록 관리 및 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []

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
            with st.spinner("과거 대화 맥락과 42개 전문 문서를 심층 분석 중..."):
                try:
                    uploaded_context, uploaded_image_part = build_uploaded_file_context(uploaded_file)

                    # [Step 1] 수동 임베딩 및 검색 (차원 불일치 에러 해결)
                    # output_dimensionality를 설정하여 3072 -> 768로 강제 조정합니다.
                    emb_res = client.models.embed_content(
                        model=EMBED_MODEL, 
                        contents=prompt,
                        config=types.EmbedContentConfig(output_dimensionality=TARGET_DIMENSION)
                    )
                    
                    search_res = index.query(
                        vector=emb_res.embeddings[0].values, 
                        top_k=15, 
                        include_metadata=True
                    )

                    context_text = ""
                    sources = set()
                    for match in search_res['matches']:
                        meta = match['metadata']
                        context_text += f"\n[출처: {meta.get('source')} (P.{meta.get('page')})]\n{meta.get('text', '')}\n---"
                        sources.add(f"{meta.get('source')} (P.{int(meta.get('page', 0))})")

                    # [Step 2] 고도화된 시스템 지침 구성
                    system_instruction = f"""
                    당신은 30년 경력의 세계 최고 시멘트 생산 및 품질 관리 기술 고문입니다. 
                    관리자와의 대화 흐름을 완벽히 파악하여, 이전 질문에서 다룬 맥락을 유지하며 답변하세요.

                    - 이번 질문에 대한 전문 문서 근거:
                    {context_text}

                    [필수 응답 지침]
                    1. **심층적 인과관계 분석**: 표면적인 현상(예: f-CaO 상승) 이면에 숨겨진 열역학적, 화학적 메커니즘을 상세히 설명하세요. 복합적인 변수들 간의 상관관계를 파악하여 기술하세요.
                    2. **풍부한 지식 활용**: 제공된 [기술 문서 내용]을 꼼꼼히 검토하여 수치, 화학식($CaO$, $C_3S$ 등), 설비 사양을 구체적으로 인용하며 신뢰도를 높이세요.
                    3. **자유롭고 상세한 서술**: 전문가가 직접 보고서를 작성하듯 논리적이고 유려하게 답변하세요. 전체적인 설명의 깊이를 최우선으로 하여 최소 1000자 이상 상세히 서술하세요.
                    4. **하이브리드 지식 결합**: 문서에 없는 내용은 실시간 웹 검색 정보를 활용하고, 당신의 공학적 추론을 결합하여 'Deep Insight'를 제공하세요.
                    5. **전문가적 제언**: 관리자가 미처 생각하지 못한 공정상의 유연성(Buffer), 설비 안정성, 원료 균일성 등의 관점에서 능동적인 조언을 아끼지 마세요.
                    """

                    # [Step 3] 대화 기록(History) 재구성
                    chat_history = []
                    for m in st.session_state.messages[:-1]:
                        role = "user" if m["role"] == "user" else "model"
                        chat_history.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))

                    # [Step 4] Gemini 3 답변 생성 (웹 검색 도구 포함)
                    google_search_tool = types.Tool(google_search=types.GoogleSearch())

                    final_user_text = f"{system_instruction}\n\n"
                    if uploaded_context:
                        final_user_text += f"[Uploaded file context]\n{uploaded_context}\n\n"
                    final_user_text += f"최종 질문: {prompt}"

                    user_parts = [types.Part(text=final_user_text)]
                    if uploaded_image_part is not None:
                        user_parts.append(uploaded_image_part)
                    
                    response = client.models.generate_content(
                        model=CHAT_MODEL,
                        contents=chat_history + [
                            types.Content(role="user", parts=user_parts)
                        ],
                        config=types.GenerateContentConfig(
                            tools=[google_search_tool], 
                            temperature=0.4 # 추론의 유연성을 위해 온도를 소폭 조정
                        )
                    )
                    
                    full_response = response.text
                    if sources:
                        full_response += "\n\n**📌 참조 문서:**\n- " + "\n- ".join(sorted(list(sources)))

                    st.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})

                except Exception as e:
                    st.error(f"⚠️ 시스템 오류 발생: {e}")

if __name__ == "__main__":
    if not st.session_state.logged_in:
        login_page()
    else:
        main_app()

