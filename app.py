import streamlit as st
import re
import io
import json
import datetime
import numpy as np
import pandas as pd
import pdfplumber
import docx
import spacy
from collections import defaultdict, Counter

from pathlib import Path
from kiwipiepy import Kiwi

import koreanize_matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from wordcloud import WordCloud

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload


# ============================================================
# 페이지 설정
# ============================================================
st.set_page_config(
    page_title="문서 전처리 시스템",
    page_icon="📄",
    layout="wide",
)

# 전역 스타일
st.markdown("""
<style>
/* 한글 웹폰트 — @import는 반드시 스타일 최상단 */
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css');
/* 디자인 토큰 — 크기·간격·폭을 한 곳에서 관리 */
:root {
    /* 타입 스케일 (16px 기준 · 비율 1.25) */
    --fs-caption: 13px;   /* 캡션 · 보조수치 */
    --fs-body:    16px;   /* 본문 · 부제 */
    --fs-label:   20px;   /* 소제목 · 라벨 */
    --fs-section: 25px;   /* 섹션 제목 · 지표값 */
    --fs-metric:  31px;   /* 지표 대제목 */
    --fs-screen:  39px;   /* 화면 대제목 */

    /* line-height — 클수록 좁게, 작을수록 넓게 */
    --lh-tight:  1.25;
    --lh-snug:   1.4;
    --lh-normal: 1.6;

    /* 간격 — 8의 배수만 */
    --space-1: 4px;
    --space-2: 8px;
    --space-3: 16px;
    --space-4: 24px;
    --space-5: 32px;
    --space-6: 48px;

    /* 폭 — 본문(measure)은 좁게, 대시보드 콘텐츠는 중간 */
    --measure: 720px;
    --content: 1200px;

    /* 상단 여백 — Streamlit 상단 헤더 */
    --gutter-top: 72px;

    /* 색 */
    --ink:     #0D2B55;
    --ink-2:   #2B3A4F;
    --muted:   #6B7A8F;
    --muted-2: #8B9AAE;
    --accent:  #4C72B0;
    --line:    #E2E9F2;
    --panel:   #F4F8FC;
    --sidebar: #F8FAFC;
}


html, body, [class*="css"] {
    font-family: 'Pretendard', 'Noto Sans KR', -apple-system, BlinkMacSystemFont, sans-serif;
    font-size: var(--fs-body);
    color: var(--ink-2);
}

/* 콘텐츠 폭 — 대시보드는 1200, 그 안에서 본문 텍스트만 720으로 가둔다 */
.block-container {
    max-width: var(--content);
    padding-top: var(--gutter-top);
    padding-bottom: var(--space-6);
}

/* 상단 헤더가 콘텐츠를 가리지 않도록 배경을 투명 처리 */
[data-testid="stHeader"] { background: transparent; }
[data-testid="stMarkdownContainer"] p {
    max-width: var(--measure);
    line-height: var(--lh-normal);
}

/* 마크다운 제목(###/####)을 타입 스케일에 맞춘다 */
h1 { font-size: var(--fs-screen)  !important; font-weight: 700 !important;
     line-height: var(--lh-tight); letter-spacing: -0.01em;
     margin: 0 0 var(--space-2) 0 !important; }
h2 { font-size: var(--fs-section) !important; font-weight: 700 !important;
     line-height: var(--lh-tight);
     margin: var(--space-4) 0 var(--space-2) 0 !important; }
h3 { font-size: var(--fs-label)   !important; font-weight: 700 !important;
     line-height: var(--lh-snug); color: var(--ink);
     margin: var(--space-4) 0 var(--space-2) 0 !important; }
h4 { font-size: var(--fs-body)    !important; font-weight: 700 !important;
     line-height: var(--lh-snug);
     margin: var(--space-3) 0 var(--space-1) 0 !important; }

/* 화면 대제목 */
.screen-title {
    font-size: var(--fs-screen); font-weight: 700; color: var(--ink);
    line-height: var(--lh-tight); letter-spacing: -0.01em;
    margin: 0 0 var(--space-1) 0; padding-top: var(--space-1);
}
.screen-sub {
    font-size: var(--fs-body); color: var(--muted);
    line-height: var(--lh-normal); max-width: var(--measure);
    margin: 0 0 var(--space-4) 0;
}

/* 지표 대제목 (탭 안) */
.metric-title {
    font-size: var(--fs-metric); font-weight: 700; color: var(--ink);
    line-height: var(--lh-tight); letter-spacing: -0.01em;
    margin: var(--space-3) 0 var(--space-1) 0; padding-top: var(--space-1);
}
.metric-sub {
    font-size: var(--fs-caption); color: var(--muted-2);
    margin: 0 0 var(--space-3) 0;
}
.metric-desc {
    font-size: var(--fs-body); color: var(--ink-2);
    background: var(--panel); border-left: 4px solid var(--accent);
    border-radius: 6px; line-height: var(--lh-normal);
    max-width: var(--measure);
    padding: var(--space-3); margin: 0 0 var(--space-4) 0;
}
/* 보조 수치 한 줄 */
.substat {
    font-size: var(--fs-caption); color: var(--muted);
    line-height: var(--lh-normal); margin-top: var(--space-2);
}

/* 사이드바 */
section[data-testid="stSidebar"] {
    min-width: 300px; background: var(--sidebar);
    border-right: 1px solid #E7EDF4;
}
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p { font-size: var(--fs-body) !important; }
section[data-testid="stSidebar"] h2 {
    font-size: var(--fs-caption) !important; font-weight: 700 !important;
    color: var(--muted); letter-spacing: 0.06em;
    margin: var(--space-4) 0 var(--space-2) 0 !important;
}

/* 사이드바 화면 선택 라디오 — 전체 폭 클릭 영역.
   width:100%가 본문의 가로 라디오까지 세로로 접어버리므로 사이드바로만 한정한다. */
section[data-testid="stSidebar"] div[role="radiogroup"] > label {
    font-size: var(--fs-body) !important;
    padding: var(--space-2); margin-bottom: var(--space-1);
    border-radius: 8px; width: 100%; transition: background 0.15s;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover { background: rgba(76,114,176,0.08); }

/* metric 카드 */
div[data-testid="stMetric"] {
    background: #F7FAFD; border: 1px solid var(--line);
    border-radius: 12px; padding: var(--space-3);
}
div[data-testid="stMetricValue"] {
    font-size: var(--fs-section) !important; font-weight: 700;
}
div[data-testid="stMetricLabel"] {
    font-size: var(--fs-caption) !important; color: #5A6B80;
}

/* 탭 · 표 · 구분선 */
button[data-baseweb="tab"] {
    font-size: var(--fs-body) !important; font-weight: 600;
    padding: var(--space-2) var(--space-3);
}
div[data-testid="stDataFrame"] { font-size: 14px; }   /* 데이터 밀도상 예외 */
hr { margin: var(--space-4) 0; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 경로 / 상수
# ============================================================
BASE_DIR = Path('.')
UNIT_DIR = BASE_DIR / 'units'
UNIT_DIR.mkdir(exist_ok=True)

UNIT_FILES = {
    'word':      UNIT_DIR / 'word.csv',
    'sentence':  UNIT_DIR / 'sentence.csv',
    'paragraph': UNIT_DIR / 'paragraph.csv',
    'document':  UNIT_DIR / 'document.csv',
}
INTEGRATED_CSV = UNIT_DIR / 'integrated.csv'
META_CSV       = UNIT_DIR / 'metadata.csv'
CUSTOM_SW_CSV  = UNIT_DIR / 'custom_stopwords.csv'

# ── 계산 결과 저장 (화면은 원본 대신 이 파일들만 읽는다) ──
#    문서가 늘어나도 화면 속도가 유지되도록 하기 위함
METRICS_CSV   = UNIT_DIR / 'metrics.csv'      # 문서별 지표
WORD_FREQ_CSV = UNIT_DIR / 'word_freq.csv'    # 작성자유형별 단어 빈도
DOC_FREQ_CSV  = UNIT_DIR / 'doc_freq.csv'     # 문서별 단어 빈도

UNITS_TO_SAVE = ['word', 'sentence', 'document']   # paragraph 잠정 보류

UNIT_COLS  = ['파일명', '작성자유형', '문서유형', '원문', '토큰']
INTEG_COLS = ['파일명', '작성자유형', '문서유형', '단위', '원문', '토큰']
META_COLS  = ['파일명', '문서유형', '작성자유형', '처리일시']

AUTHOR_TYPES = ['Human', 'AI']
DOC_TYPES    = ['report', 'essay']     # 드라이브에 폴더를 만들면 자동 추가됨
DATA_FOLDER  = 'DATA'                   

MASTER_SHEET_NAME = '문서분석_마스터시트'

# ── 감성사전 ──
#    KNU를 기본으로 고정한다. IBA_DCX는 참고용 비교 기능으로만 제공한다.
SENTI_DICTS = {
    'KNU (범용 한국어)':          'KNU_SentiWord_Dict.txt',
    'IBA_DCX (음식점 리뷰 특화)': 'IBA_DCX_Dictionary.csv',
}
SENTI_DEFAULT = 'KNU (범용 한국어)'

# 감성 범주 — 야오즈옌 외(2024) 기준인 긍정·부정 2범주로 고정
SENTI_N_CAT = 2

# LDA 기본 설정
LDA_N_TOPICS    = 5
LDA_MIN_DOCS    = 30      # 이 미만이면 내용 다양성 신뢰도 경고
LDA_RANDOM_SEED = 42      # 재현성 확보

# 단어 조회 화면에서 기본 표시할 주요 단어
# 후속 연구에서 이 목록만 교체하면 됨 (현재는 임의 선정)
KEY_WORDS = ['분석', '연구', '결과', '방법', '데이터',
             '중요', '필요', '문제', '효과', '영향']

# ── 기초 통계량 ──
S_NSENT = '문장 수'
S_NWORD = '단어 수'
S_NUNIQ = '고유 단어 수'
S_SLEN  = '평균 문장 길이'
S_SVAR  = '문장 길이 분산'
BASIC_COLS = [S_NSENT, S_NWORD, S_NUNIQ, S_SLEN, S_SVAR]

# ── 문서 특성 지표 (Diversity) ──
D_LEX = '어휘 다양성'
D_CON = '내용 다양성'
D_SEN = '감성 다양성'
D_BUR = '출현 편중도'
DIV_COLS = [D_LEX, D_CON, D_SEN, D_BUR]


# ── 지표 열 목록 ──
METRICS_COLS = (
    ["파일명", "작성자유형", "문서유형"]
    + BASIC_COLS
    + DIV_COLS
    + [
        "감성매칭수",
        "감성매칭률",
        "감성사전",
    ]
)

WORD_FREQ_COLS = ["작성자유형", "토큰", "빈도"]
DOC_FREQ_COLS = ["파일명", "토큰", "빈도"]

METRIC_EN = {
    D_LEX: 'Lexical Diversity (Type-Token Ratio)',
    D_CON: 'Content Diversity (Rao-Stirling Index)',
    D_SEN: 'Emotional Diversity (Shannon Entropy)',
    D_BUR: 'Burstiness',
    S_NSENT: 'Number of Sentences',
    S_NWORD: 'Number of Tokens',
    S_NUNIQ: 'Number of Types',
    S_SLEN: 'Mean Sentence Length',
    S_SVAR: 'Sentence Length Variance',
}

METRIC_DESC = {
    S_NSENT: '문서를 구성하는 문장의 개수',
    S_NWORD: '형태소 분석으로 분리된 전체 단어의 개수 (중복 포함)',
    S_NUNIQ: '서로 다른 단어의 개수 (중복 제외)',
    S_SLEN:  '문장 하나에 들어 있는 단어의 평균 개수',
    S_SVAR:  '문장별 단어 수가 평균 문장 길이에서 얼마나 차이 나는지를 나타내는 값 '
             '(값이 작을수록 비슷한 길이의 문장이 이어짐)',
}

# ── 다양성 지표 상세 설명 ──
# 대시보드와 PDF 보고서에서 공통으로 사용한다.
DIVERSITY_INFO = {
    D_LEX: {
        "definition": (
            "문서에서 사용된 전체 단어(Token) 대비 서로 다른 단어(Type)의 "
            "비율을 나타내는 지표로, 어휘 사용의 다양성을 측정한다."
        ),
        "formulas": [
            {
                "label": "",
                "latex": r"\mathrm{TTR} = \frac{V}{N}",
            }
        ],
        "variables": [
            ("V", "고유 단어(Type)의 수"),
            ("N", "전체 단어(Token)의 수"),
        ],
        "interpretation": (
            "값이 높을수록 동일한 단어의 반복이 적고 다양한 어휘를 "
            "사용하는 문서로 해석된다."
        ),
    },

    D_CON: {
        "definition": (
            "토픽 모델링을 통해 추출된 주제의 분포와 주제 간 차이를 "
            "종합하여 문서의 내용적 다양성을 측정하는 지표이다."
        ),
        "formulas": [
            {
                "label": "",
                # \substack 대신 호환성이 높은 표현 사용
                "latex": r"\mathrm{CD} = \sum_{i \ne j} p_i p_j d_{ij}",
            }
        ],
        "variables": [
            ("pᵢ", "토픽 i의 출현 비율"),
            ("pⱼ", "토픽 j의 출현 비율"),
            ("dᵢⱼ", "토픽 i와 j 간 거리(Cosine distance)"),
        ],
        "interpretation": (
            "값이 높을수록 다양한 주제를 균형 있게 포함하며, 서로 다른 "
            "내용 요소를 폭넓게 다루는 문서로 해석된다."
        ),
    },

    D_SEN: {
        "definition": (
            "감성 범주의 분포를 Shannon Entropy로 계산하여 문서에 "
            "표현된 감성의 다양성과 균형 정도를 측정하는 지표이다."
        ),
        "formulas": [
            {
                "label": "",
                "latex": (
                    r"\mathrm{ED} = "
                    r"-\sum_{i=1}^{n} p_i \log_{2}(p_i)"
                ),
            }
        ],
        "variables": [
            ("pᵢ", "감성 범주 i의 출현 비율"),
            ("n", "감성 범주의 수"),
        ],
        "interpretation": (
            "값이 높을수록 다양한 감성이 균형 있게 표현된 문서로 해석된다."
        ),
        "note": (
            "현재 계산에서는 긍정·부정의 2개 감성 범주와 "
            "밑이 2인 로그(log₂)를 사용한다."
        ),
    },

    D_BUR: {
        "definition": (
            "문서 내에서 반복 출현하는 단어가 특정 구간에 집중되어 "
            "나타나는 정도를 측정하는 지표이다. 문서에서 3회 이상 등장한 "
            "각 단어의 출현 간격을 이용하여 계산한 후 평균값을 사용한다."
        ),
        "formulas": [
            {
                "label": "단어 w의 편중도",
                "latex": (
                    r"B_w = "
                    r"\frac{\sigma_w - \mu_w}"
                    r"{\sigma_w + \mu_w}"
                ),
            },
            {
                "label": "문서의 출현 편중도",
                "latex": (
                    r"\mathrm{Burstiness} = "
                    r"\frac{1}{|W|}"
                    r"\sum_{w \in W} B_w"
                ),
            },
        ],
        "variables": [
            ("μ₍w₎", "단어 w의 출현 간격 평균"),
            ("σ₍w₎", "단어 w의 출현 간격 표준편차"),
            ("W", "문서에서 3회 이상 등장한 단어 집합"),
            ("|W|", "편중도 계산에 포함된 단어의 수"),
        ],
        "interpretation": (
            "값이 높을수록 반복 단어가 특정 구간에 집중되어 나타나며, "
            "값이 낮을수록 반복 단어가 문서 전반에 비교적 고르게 "
            "분포한 것으로 해석된다."
        ),
    },
}

def render_diversity_description(metric):
    """다양성 지표 설명을 대시보드에 표시한다."""
    info = DIVERSITY_INFO.get(metric)

    if info is None:
        st.warning("해당 지표의 설명 정보가 없습니다.")
        return

    st.markdown("#### 정의")
    st.write(info["definition"])

    st.markdown("#### 수식")

    for formula in info["formulas"]:
        label = formula.get("label", "")

        if label:
            st.markdown(f"**{label}**")

        # LaTeX 수식은 반드시 st.latex()로 별도 출력
        st.latex(formula["latex"])

    st.markdown("#### 기호 설명")

    for symbol, description in info["variables"]:
        # 기호만 LaTeX로 렌더링하지 않고 일반 텍스트로 출력하여
        # Markdown-LaTeX 충돌을 방지한다.
        clean_symbol = (
            symbol
            .replace(r"\mu", "μ")
            .replace(r"\sigma", "σ")
            .replace("_w", "₍w₎")
        )

        st.markdown(f"- **{clean_symbol}**: {description}")

    st.markdown("#### 해석")
    st.write(info["interpretation"])

    if info.get("note"):
        st.caption(info["note"])

# 색상 (대시보드 그래프)
C_TARGET = '#DD8452'      # 현재 문서
C_HUMAN  = '#4C72B0'
C_AI     = '#C44E52'


# ── 드라이브 동기화 대상 ──
SYNC_FILES = {
    'word.csv':             UNIT_FILES['word'],
    'sentence.csv':         UNIT_FILES['sentence'],
    'document.csv':         UNIT_FILES['document'],
    'integrated.csv':       INTEGRATED_CSV,
    'metadata.csv':         META_CSV,
    'custom_stopwords.csv': CUSTOM_SW_CSV,
    'metrics.csv':          METRICS_CSV,
    'word_freq.csv':        WORD_FREQ_CSV,
    'doc_freq.csv':         DOC_FREQ_CSV,
}


def init_local_csvs():
    for name in ['word', 'sentence', 'document']:
        if not UNIT_FILES[name].exists():
            pd.DataFrame(columns=UNIT_COLS).to_csv(
                UNIT_FILES[name], index=False, encoding='utf-8-sig')
    if not INTEGRATED_CSV.exists():
        pd.DataFrame(columns=INTEG_COLS).to_csv(
            INTEGRATED_CSV, index=False, encoding='utf-8-sig')
    if not META_CSV.exists():
        pd.DataFrame(columns=META_COLS).to_csv(
            META_CSV, index=False, encoding='utf-8-sig')
    if not CUSTOM_SW_CSV.exists():
        pd.DataFrame(columns=['불용어']).to_csv(
            CUSTOM_SW_CSV, index=False, encoding='utf-8-sig')


# ============================================================
# 워드클라우드 전용 불용어
# ============================================================
WC_STOPWORDS = {
    '이', '그', '저', '것', '거', '곳', '수', '좀', '처럼', '까지', '에도', '에도요',
    '이나', '라도',
    '그리고', '그래서', '그러나', '하지만', '또한', '즉', '결국', '때문에', '그래도',
    '합니다', '해요', '했어요', '하네요', '하시네요', '하시던데요', '같아요', '있어요',
    '없어요', '되네요', '되었어요', '보여요', '느껴져요', '하겠습니다', '되겠습니다',
    '있습니다', '없습니다', '이에요', '이라', '해서',
    'ㅎㅎ', 'ㅋㅋ', 'ㅠㅠ', '^^', '^^;;', '~', '~~', '!!!', '??', '!?', '?!', '...',
    '!!', '~!!', '~^^!!',
    '아주', '정말', '진짜', '엄청', '매우', '완전', '너무', '굉장히', '많이', '많아요',
    '적당히', '넘',
    '정도', '느낌', '같은', '니당', '네요', '있네요', '이네요', '이라서',
    '해서요', '보니까', '봤어요', '먹었어요', '마셨어요', '갔어요', '봤습니다', '하는',
    '하게', '드네', '또시', '이랑', '하고', '해도', '해도요', '때문에요', '이나요',
    '정도에요',
    '때', '점', '중', '등', '분', '개', '명', '번', '내', '외', '및', '경우', '통해',
    '위해', '대한', '대해', '관련', '이런', '저런', '그런', '어떤', '무엇', '누구',
    '하다', '있다', '없다', '되다', '이다', '아니다', '같다', '보다', '싶다', '주다',
    '가다', '오다', '만들다', '들다', '나다', '받다', '통하다', '위하다', '대하다',
    '나', '너', '우리', '저희', '자기', '그것', '이것', '저것', '여기', '거기', '저기',
    '누가', '무슨',
    '더', '또', '즉', '및', '단', '약', '매우', '가장', '거의', '다시', '함께', '바로',
    '먼저', '이제', '아직', '항상', '자주', '특히', '역시', '물론', '오히려',
    '은', '는', '이', '가', '을', '를', '의', '에', '에서', '와', '과', '도', '만',
    '로', '으로', '에게', '한테', '께', '보다', '부터', '마다', '조차', '마저',
    'ㅂ니다', 'ㅆ다', '다', '요', '고', '며', '면', '서', '자',
    '데', '지', '든', '든지', '나마',
    # 단독 자모(ㄴ/ㄹ/ㅁ 등)는 is_lone_jamo()가 별도로 제거함
}


# ============================================================
# 모델 / 감성사전 초기화 (캐싱)
# ============================================================
@st.cache_resource
def load_models():
    kiwi   = Kiwi()
    nlp_en = spacy.load('en_core_web_sm')
    kfont  = font_manager.findfont(plt.rcParams['font.family'][0])
    return kiwi, nlp_en, kfont

kiwi, nlp_en, KFONT_PATH = load_models()

# 한글 폰트에는 유니코드 마이너스가 없어 그래프에서 깨진다 → 하이픈 사용
plt.rcParams['axes.unicode_minus'] = False


@st.cache_data
def load_sentiment_dict(dict_name):
    """
    지정한 감성사전 → ({token: score}, 오류메시지)
    · TXT (KNU): 탭 구분 '단어\\t점수', 구(phrase) 항목은 제외
    · CSV (IBA_DCX): 컬럼 자동 감지
    """
    fname = SENTI_DICTS[dict_name]
    p = BASE_DIR / fname
    if not p.exists():
        return None, f"'{fname}' 파일을 app.py와 같은 폴더에서 찾지 못했습니다."

    if p.suffix.lower() == '.txt':
        d = {}
        try:
            with open(p, encoding='utf-8') as f:
                for line in f:
                    line = line.rstrip('\n')
                    if not line.strip() or '\t' not in line:
                        continue
                    word, score = line.rsplit('\t', 1)
                    word = word.strip()
                    if ' ' in word:      # 구(phrase) 제외 — 형태소 토큰과 매칭 불가
                        continue
                    try:
                        d[word] = float(score)
                    except ValueError:
                        continue
        except Exception as e:
            return None, f"사전을 읽는 중 오류: {e}"
        return (d, None) if d else (None, "사전에서 유효한 항목을 찾지 못했습니다.")

    try:
        df = pd.read_csv(p)
    except Exception as e:
        return None, f"사전을 읽는 중 오류: {e}"

    token_col = next((c for c in ['token', 'Token', 'word', 'Word', '단어', '토큰']
                      if c in df.columns), None)
    score_col = next((c for c in ['expert_adjusted', 'expert_adj', 'score', 'Score',
                                  'polarity', 'sentiment', '감성점수', '점수']
                      if c in df.columns), None)
    if token_col is None or score_col is None:
        return None, f"토큰/점수 열을 찾지 못했습니다. 현재 열: {list(df.columns)}"

    sub = df[[token_col, score_col]].dropna()
    try:
        sub[score_col] = sub[score_col].astype(float)
    except ValueError:
        return None, f"점수 열 '{score_col}'을 숫자로 변환할 수 없습니다."
    return dict(zip(sub[token_col].astype(str), sub[score_col])), None


# ============================================================
# 구글 드라이브 · 구글 시트 연동
# ============================================================
# 주의: 구글 시트를 쓰려면 spreadsheets 권한이 필요하다.
#       권한 범위가 바뀌면 기존 토큰이 무효가 되므로 재로그인이 필요하다.
SCOPES = [
    'https://www.googleapis.com/auth/drive',
    'https://www.googleapis.com/auth/spreadsheets',
]

# OAuth 인증 파일 경로
#  · credentials.json : 구글 클라우드에서 받은 OAuth 클라이언트 ID 파일 (직접 준비)
#  · token.json       : 최초 로그인 후 자동 생성되는 인증 토큰 (자동 생성)
CREDENTIALS_FILE = BASE_DIR / 'credentials.json'
TOKEN_FILE       = BASE_DIR / 'token.json'


@st.cache_resource
def get_credentials():
    """
    OAuth 인증 정보 획득.
    · Streamlit Secrets에 토큰이 있으면 그것을 사용 (클라우드 배포)
    · 없으면 로컬 token.json 사용, 그마저 없으면 브라우저 로그인 (로컬 실행)
    """
    cloud = "gcp_oauth_token" in st.secrets

    if cloud:
        creds = Credentials.from_authorized_user_info(
            dict(st.secrets["gcp_oauth_token"]), SCOPES)
    elif TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
    else:
        creds = None

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            creds = InstalledAppFlow.from_client_secrets_file(
                str(CREDENTIALS_FILE), SCOPES).run_local_server(port=0)
        if not cloud:                       # 클라우드에서는 파일 저장 불필요
            TOKEN_FILE.write_text(creds.to_json(), encoding='utf-8')

    return creds


@st.cache_resource
def get_drive_service():
    return build('drive', 'v3', credentials=get_credentials())


@st.cache_resource
def get_sheets_service():
    return build('sheets', 'v4', credentials=get_credentials())


def granted_scopes():
    """
    실제 토큰에 부여된 권한 범위를 읽는다.
    Credentials 객체의 .scopes 는 '요청한' 범위를 그대로 돌려주므로
    예전 토큰이어도 권한이 있는 것처럼 보인다. 원본을 직접 확인해야 한다.
    """
    try:
        if "gcp_oauth_token" in st.secrets:
            return list(st.secrets["gcp_oauth_token"].get("scopes", []))
        if TOKEN_FILE.exists():
            return json.loads(TOKEN_FILE.read_text(encoding='utf-8')).get('scopes', [])
    except Exception:
        pass
    return []


def has_sheets_permission():
    """토큰에 스프레드시트 권한이 실제로 포함되어 있는지 확인"""
    return any('spreadsheets' in s for s in granted_scopes())


# ── 드라이브 기본 조작 ──
def drive_find_child(service, parent_id, name, is_folder=True):
    mime = " and mimeType='application/vnd.google-apps.folder'" if is_folder else ""
    q = f"'{parent_id}' in parents and name='{name}'{mime} and trashed=false"
    res = service.files().list(
        q=q, fields="files(id, name)",
        supportsAllDrives=True, includeItemsFromAllDrives=True,
    ).execute()
    files = res.get('files', [])
    return files[0]['id'] if files else None


def drive_ensure_folder(service, parent_id, name):
    fid = drive_find_child(service, parent_id, name, is_folder=True)
    if fid:
        return fid
    meta = {'name': name,
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [parent_id]}
    folder = service.files().create(
        body=meta, fields='id', supportsAllDrives=True).execute()
    return folder['id']


def drive_list_subfolders(service, parent_id):
    q = (f"'{parent_id}' in parents and trashed=false "
         f"and mimeType='application/vnd.google-apps.folder'")
    res = service.files().list(
        q=q, fields="files(id, name)", pageSize=200,
        supportsAllDrives=True, includeItemsFromAllDrives=True,
    ).execute()
    return res.get('files', [])


def drive_list_pdfs(service, folder_id):
    q = (f"'{folder_id}' in parents and trashed=false "
         f"and mimeType='application/pdf'")
    res = service.files().list(
        q=q, fields="files(id, name)", pageSize=1000,
        supportsAllDrives=True, includeItemsFromAllDrives=True,
    ).execute()
    return res.get('files', [])


def drive_download_bytes(service, file_id):
    request    = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    buf        = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    buf.seek(0)
    return buf.read()


def drive_upload_bytes(service, folder_id, filename, data, mimetype):
    """바이트를 드라이브에 업로드 (동일 이름 있으면 갱신)"""
    media = MediaIoBaseUpload(io.BytesIO(data), mimetype=mimetype, resumable=False)
    existing = drive_find_child(service, folder_id, filename, is_folder=False)
    if existing:
        service.files().update(fileId=existing, media_body=media,
                               supportsAllDrives=True).execute()
    else:
        service.files().create(body={'name': filename, 'parents': [folder_id]},
                               media_body=media, fields='id',
                               supportsAllDrives=True).execute()


def drive_upload_csv(service, folder_id, filename, local_path):
    drive_upload_bytes(service, folder_id, filename,
                       local_path.read_bytes(), 'text/csv')


def folder_url(folder_id):
    return f"https://drive.google.com/drive/folders/{folder_id}"


def sheet_url(spreadsheet_id):
    return f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}"


# ── 폴더 구조 ──
def get_folder_structure():
    """
    루트 아래 DATA/작성자유형/문서유형 폴더와 output 폴더를 확보한다.
    문서유형은 기본값(DOC_TYPES)에 더해 드라이브에 실제로 존재하는
    하위 폴더까지 자동으로 인식한다. → 폴더를 만들면 곧 새 문서유형이 된다.
    """
    service = get_drive_service()
    root_id = st.secrets["drive"]["root_folder_id"]
    data_id = drive_ensure_folder(service, root_id, DATA_FOLDER)   # 입력 폴더를 감싸는 DATA
    folders = {'root': root_id,
               'data': data_id,
               'output': drive_ensure_folder(service, root_id, 'output')}

    doc_types = set(DOC_TYPES)
    for author in AUTHOR_TYPES:
        author_id = drive_ensure_folder(service, data_id, author)   # DATA 아래에 생성
        folders[author] = author_id
        for f in drive_list_subfolders(service, author_id):
            doc_types.add(f['name'])

    doc_types = sorted(doc_types)
    for author in AUTHOR_TYPES:
        for doc in doc_types:
            folders[(author, doc)] = drive_ensure_folder(
                service, folders[author], doc)
    return service, folders, doc_types


def sync_from_drive(service, output_id):
    for fname, local_path in SYNC_FILES.items():
        fid = drive_find_child(service, output_id, fname, is_folder=False)
        if fid:
            local_path.write_bytes(drive_download_bytes(service, fid))
    init_local_csvs()


def sync_to_drive(service, output_id):
    for fname, local_path in SYNC_FILES.items():
        if local_path.exists():
            drive_upload_csv(service, output_id, fname, local_path)


# ── 구글 시트 마스터 시트 ──
def ensure_master_sheet(service, output_id):
    """output 폴더에 마스터 시트를 찾거나 새로 만들고 ID 반환"""
    q = (f"'{output_id}' in parents and name='{MASTER_SHEET_NAME}' "
         f"and mimeType='application/vnd.google-apps.spreadsheet' "
         f"and trashed=false")
    res = service.files().list(q=q, fields="files(id, name)",
                               supportsAllDrives=True,
                               includeItemsFromAllDrives=True).execute()
    files = res.get('files', [])
    if files:
        return files[0]['id']
    meta = {'name': MASTER_SHEET_NAME,
            'mimeType': 'application/vnd.google-apps.spreadsheet',
            'parents': [output_id]}
    created = service.files().create(body=meta, fields='id',
                                     supportsAllDrives=True).execute()
    return created['id']


def write_master_sheet(sheets_service, spreadsheet_id, df):
    """
    마스터 시트를 df 내용으로 갱신한다.
    문서 수가 많지 않으므로(수백 건) 전체를 다시 쓰는 방식이 가장 안전하다.
    """
    values = [list(df.columns)] + df.astype(object).where(
        pd.notna(df), '').values.tolist()
    sheets_service.spreadsheets().values().clear(
        spreadsheetId=spreadsheet_id, range='A:ZZ', body={}).execute()
    sheets_service.spreadsheets().values().update(
        spreadsheetId=spreadsheet_id, range='A1',
        valueInputOption='RAW', body={'values': values}).execute()


# ============================================================
# 텍스트 추출
# ============================================================
def extract_text_from_bytes(file_bytes, suffix='.pdf'):
    if suffix == '.pdf':
        texts = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                bboxes = [t.bbox for t in page.find_tables()]
                if bboxes:
                    p = page
                    for bbox in bboxes:
                        p = p.outside_bbox(bbox)
                    text = p.extract_text()
                else:
                    text = page.extract_text()
                if text:
                    texts.append(text)
        return '\n'.join(texts)

    elif suffix == '.docx':
        doc   = docx.Document(io.BytesIO(file_bytes))
        texts = []
        for block in doc.element.body:
            if block.tag.endswith('}p'):
                para = docx.text.paragraph.Paragraph(block, doc)
                if para.text.strip():
                    texts.append(para.text)
        return '\n'.join(texts)

    else:
        raise ValueError(f"지원하지 않는 형식: {suffix}")


# ============================================================
# 기본 전처리
# ============================================================
def remove_html_tags(text):
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'&[a-zA-Z0-9#]+;', '', text)
    return text

def remove_urls(text):
    for p in [r'https?://\S+', r'www\.\S+',
              r'\S+\.com\S*', r'\S+\.co\.kr\S*']:
        text = re.sub(p, '', text, flags=re.IGNORECASE)
    return text

def clean_special_characters(text):
    text = re.sub(r'^\s*\d{1,4}\s*$', '', text, flags=re.MULTILINE)
    for char in ['\xa0', '\x00', '\r', '\ufeff']:
        text = text.replace(char, ' ')
    text = re.sub(r'["""]', '"', text)
    text = re.sub(r"[''']", "'", text)
    text = re.sub(r'[『』「」]', '"', text)
    text = re.sub(r'[〔〕【】]',
                  lambda m: '[' if m.group() in '〔【' else ']', text)
    text = re.sub(r'…', '...', text)
    return text

def normalize_whitespace(text):
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def preprocess_basic(text):
    text = remove_html_tags(text)
    text = remove_urls(text)
    text = clean_special_characters(text)
    text = normalize_whitespace(text)
    return text


# ============================================================
# 토큰화
# ============================================================
def lemmatize_english(word):
    doc = nlp_en(word)
    return doc[0].lemma_ if doc else word

def tokenize(text):
    tokens       = kiwi.tokenize(text)
    result       = []
    total        = len(tokens)
    skip_indices = set()

    for i, token in enumerate(tokens):
        if i in skip_indices:
            continue

        word = token.form
        pos  = str(token.tag)

        if pos == 'SL':
            lemma = lemmatize_english(word)
            result.append((word, pos, 'en', lemma))

        elif pos == 'SH':
            result.append((word, pos, 'zh', word))

        elif pos in ['VA', 'VV', 'VX', 'VCN', 'VCP']:
            base   = word if word.endswith('다') else word + '다'
            has_ep = False

            next_eomi     = None
            next_eomi_idx = None
            for j in range(i + 1, min(i + 4, total)):
                next_tag = str(tokens[j].tag)
                if next_tag == 'EP':
                    has_ep = True
                    continue
                if next_tag in ['EF', 'EC', 'ETM', 'ETN']:
                    next_eomi     = next_tag
                    next_eomi_idx = j
                break

            if (not has_ep
                    and next_eomi == 'EF'
                    and next_eomi_idx is not None
                    and tokens[next_eomi_idx].form == '다'
                    and not word.endswith('다')):
                skip_indices.add(next_eomi_idx)

            if next_eomi == 'EF':
                result.append((base, pos, 'ko', base))
            else:
                result.append((word, pos, 'ko', base))

        else:
            result.append((word, pos, 'ko', word))

    result = [(w, p, l, b) for w, p, l, b in result if w.strip() and b.strip()]
    return result


# ============================================================
# 다단위 분리
# ============================================================
def split_paragraphs(text, min_chars=50):
    """PDF 문단 휴리스틱 (결과 부정확 → paragraph 저장은 보류 상태)"""
    merged = re.sub(r'\n', '', text)
    merged = re.sub(r'([다요음까네]\.)\s*', r'\1\n', merged)
    sents = [s.text.strip() for s in kiwi.split_into_sents(merged)
             if s.text.strip()]
    paragraphs, buf = [], ""
    for sent in sents:
        buf = (buf + " " + sent).strip()
        if len(buf) >= min_chars:
            paragraphs.append(buf)
            buf = ""
    if buf:
        if paragraphs:
            paragraphs[-1] += " " + buf
        else:
            paragraphs.append(buf)
    return paragraphs

def split_sentences(text):
    return [s.text for s in kiwi.split_into_sents(text)]

def to_token_str(text):
    return ' '.join(t[0] for t in tokenize(preprocess_basic(text)))

def split_units(raw_text):
    units = {'word': [], 'sentence': [], 'paragraph': [], 'document': []}

    doc_clean = preprocess_basic(raw_text)
    units['document'].append((doc_clean, to_token_str(raw_text)))

    paragraphs = split_paragraphs(raw_text)
    for para in paragraphs:
        units['paragraph'].append((para, to_token_str(para)))

    for para in paragraphs:
        for sent in split_sentences(para):
            sent = sent.strip()
            if sent:
                units['sentence'].append((sent, to_token_str(sent)))

    for form, tag, lang, base in tokenize(doc_clean):
        if not form.strip() or not base.strip():
            continue
        units['word'].append((form, base))

    return units


# ============================================================
# 처리 / 저장 / 관리
# ============================================================
def process_document(fname, file_bytes, doc_type, author_type):
    try:
        raw   = extract_text_from_bytes(file_bytes, '.pdf')
        units = split_units(raw)

        for unit_name in UNITS_TO_SAVE:
            rows = [[fname, author_type, doc_type, rawtxt, tok]
                    for rawtxt, tok in units[unit_name]]
            new = pd.DataFrame(rows, columns=UNIT_COLS)
            old = pd.read_csv(UNIT_FILES[unit_name], encoding='utf-8-sig')
            pd.concat([old, new], ignore_index=True).to_csv(
                UNIT_FILES[unit_name], index=False, encoding='utf-8-sig')

        integrated_rows = []
        for unit_name in UNITS_TO_SAVE:
            for rawtxt, tok in units[unit_name]:
                integrated_rows.append(
                    [fname, author_type, doc_type, unit_name, rawtxt, tok])
        new_int = pd.DataFrame(integrated_rows, columns=INTEG_COLS)
        old_int = pd.read_csv(INTEGRATED_CSV, encoding='utf-8-sig')
        pd.concat([old_int, new_int], ignore_index=True).to_csv(
            INTEGRATED_CSV, index=False, encoding='utf-8-sig')

        meta = pd.read_csv(META_CSV, encoding='utf-8-sig')
        new_meta = pd.DataFrame([[
            fname, doc_type, author_type,
            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ]], columns=META_COLS)
        pd.concat([meta, new_meta], ignore_index=True).to_csv(
            META_CSV, index=False, encoding='utf-8-sig')

        counts = {u: len(units[u]) for u in UNITS_TO_SAVE}
        return counts, None

    except Exception as e:
        return None, str(e)


def reset_all(service, output_id):
    """처리 결과만 비운다. 사용자가 추가한 불용어는 설정값이므로 유지한다."""
    for name in ['word', 'sentence', 'document']:
        pd.DataFrame(columns=UNIT_COLS).to_csv(
            UNIT_FILES[name], index=False, encoding='utf-8-sig')
    pd.DataFrame(columns=INTEG_COLS).to_csv(
        INTEGRATED_CSV, index=False, encoding='utf-8-sig')
    pd.DataFrame(columns=META_COLS).to_csv(
        META_CSV, index=False, encoding='utf-8-sig')
    pd.DataFrame(columns=METRICS_COLS).to_csv(
    METRICS_CSV,
    index=False,
    encoding="utf-8-sig",
    )
    pd.DataFrame(columns=WORD_FREQ_COLS).to_csv(
        WORD_FREQ_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(columns=DOC_FREQ_COLS).to_csv(
        DOC_FREQ_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    NOTE_FILE.write_text("", encoding="utf-8")

    sync_to_drive(service, output_id)

    if has_sheets_permission():
        sheet_id = ensure_master_sheet(service, output_id)

        empty_master = pd.DataFrame(columns=METRICS_COLS)

        write_master_sheet(
            get_sheets_service(),
            sheet_id,
            empty_master,
        )


# ── 사용자 추가 불용어 ──
def load_custom_stopwords():
    if not CUSTOM_SW_CSV.exists():
        return set()
    df = pd.read_csv(CUSTOM_SW_CSV, encoding='utf-8-sig')
    return set(df['불용어'].dropna().astype(str))


def save_custom_stopwords(words):
    pd.DataFrame({'불용어': sorted(words)}).to_csv(
        CUSTOM_SW_CSV, index=False, encoding='utf-8-sig')
    drive_upload_csv(service, OUTPUT_ID, 'custom_stopwords.csv', CUSTOM_SW_CSV)


# ============================================================
# 토큰 정리
# ============================================================
def is_lone_jamo(tok):
    """
    Kiwi 분석 부산물인 단독 자모 여부.
    · 호환용 자모  ㄱ~ㅎ, ㅏ~ㅣ : U+3131~U+3163
    · 한글 자모 초/중/종성       : U+1100~U+11FF (예: 종성 ᆫ U+11AB)
    """
    if len(tok) != 1:
        return False
    cp = ord(tok)
    return (0x3131 <= cp <= 0x3163) or (0x1100 <= cp <= 0x11FF)


# ============================================================
# 지표 계산 — 기초 통계량
# ============================================================
def sent_token_count(tok_str):
    """문장 토큰 문자열의 공백 기준 단어 수"""
    if not isinstance(tok_str, str):
        return 0
    t = tok_str.strip()
    return len(t.split()) if t else 0


# ============================================================
# 지표 계산 — Diversity
# ============================================================
def compute_lexical_diversity(tokens):
    """어휘 다양성 (TTR) = 고유 단어 수 / 전체 단어 수"""
    n = len(tokens)
    return (len(set(tokens)) / n) if n else np.nan


def compute_burstiness(tokens):
    """
    출현 편중도 = (표준편차-평균)/(표준편차+평균)   (-1: 규칙적 ~ +1: 편중)
    문서 내 3회 이상 나온 단어만 사용(간격 2개 이상 확보), 그 평균을 반환
    """
    pos = defaultdict(list)
    for i, t in enumerate(tokens):
        pos[t].append(i)
    bs = []
    for t, idxs in pos.items():
        if len(idxs) < 3:
            continue
        gaps = np.diff(idxs)
        mu, sd = gaps.mean(), gaps.std()
        if mu + sd > 0:
            bs.append((sd - mu) / (sd + mu))
    return float(np.mean(bs)) if bs else np.nan


def categorize_sentiment(score, n_cat=SENTI_N_CAT):
    """감성 점수를 범주로 변환. 2범주에서는 중립(0)을 제외한다."""
    if n_cat == 2:
        if score > 0:  return '긍정'
        if score < 0:  return '부정'
        return None
    if n_cat == 3:
        if score > 0:  return '긍정'
        if score < 0:  return '부정'
        return '중립'
    return f'{int(score):+d}'


def compute_sentiment_diversity(tokens, senti, n_cat=SENTI_N_CAT):
    """
    감성 다양성 = Shannon 엔트로피 (밑 2)
        H = - Σ p(t_i) · log2 p(t_i)
    야오즈옌 외(2024) Equation (2) 적용
    반환: (엔트로피, 매칭 단어 수, 매칭률 %)
    """
    n = len(tokens)
    if not senti or n == 0:
        return np.nan, 0, 0.0

    cats = []
    for t in tokens:
        if t in senti:
            c = categorize_sentiment(senti[t], n_cat)
            if c is not None:
                cats.append(c)

    matched = len(cats)
    rate = (matched / n * 100) if n else 0.0
    if matched < 2:
        return np.nan, matched, rate

    counts = Counter(cats)
    p = np.array([c / matched for c in counts.values()], dtype=float)
    return max(float(-(p * np.log2(p)).sum()), 0.0), matched, rate


def compute_sentiment_diversity_from_freq(tok_counts, senti, n_cat=SENTI_N_CAT):
    """
    (토큰, 빈도) 쌍으로 감성 다양성을 계산한다.
    감성 다양성은 단어의 순서와 무관하므로 집계 자료만으로 계산할 수 있다.
    → 원본을 읽지 않아도 되어 문서가 많을 때 유리하다.
    """
    total = sum(c for _, c in tok_counts)
    if not senti or total == 0:
        return np.nan, 0, 0.0
    cats, matched = Counter(), 0
    for t, c in tok_counts:
        if t in senti:
            k = categorize_sentiment(senti[t], n_cat)
            if k is not None:
                cats[k] += c
                matched += c
    rate = matched / total * 100
    if matched < 2:
        return np.nan, matched, rate
    p = np.array([v / matched for v in cats.values()], dtype=float)
    return max(float(-(p * np.log2(p)).sum()), 0.0), matched, rate


def compute_content_diversity(doc_texts, n_topics=LDA_N_TOPICS,
                              seed=LDA_RANDOM_SEED):
    """
    내용 다양성 = Rao-Stirling 지수
        D = Σ p_i · p_j · d_ij   (i ≠ j)
    · p_i  : 문서 내 토픽 i의 비중 (LDA)
    · d_ij : 토픽 i·j의 코사인 거리 (토픽-단어 행렬)
    야오즈옌 외(2024) Equation (1) 적용

    doc_texts : [(파일명, 토큰 문자열), ...]
    반환      : ({파일명: 값}, 안내메시지)
    """
    names = [n for n, _ in doc_texts]
    texts = [t for _, t in doc_texts]
    if len(texts) < 2:
        return {}, "문서가 2건 이상이어야 계산할 수 있습니다."

    try:
        vec = CountVectorizer(tokenizer=str.split, lowercase=False,
                              token_pattern=None, min_df=1)
        X = vec.fit_transform(texts)
        if X.shape[1] < n_topics:
            return {}, (f"어휘 수({X.shape[1]})가 주제 수({n_topics})보다 적어 "
                        f"계산할 수 없습니다.")

        lda = LatentDirichletAllocation(
            n_components=n_topics, random_state=seed,
            learning_method='batch', max_iter=50)
        doc_topic = lda.fit_transform(X)
        row_sum = doc_topic.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        doc_topic = doc_topic / row_sum                     # p_i

        tw = lda.components_ / lda.components_.sum(axis=1, keepdims=True)
        d = 1.0 - cosine_similarity(tw)                     # d_ij
        np.fill_diagonal(d, 0.0)
        d = np.clip(d, 0.0, None)

        # d의 대각이 0이므로 p·d·p 가 곧 Σ_{i≠j} p_i p_j d_ij 와 같다
        return {name: float(p @ d @ p)
                for name, p in zip(names, doc_topic)}, None

    except Exception as e:
        return {}, f"계산 중 오류가 발생했습니다: {e}"


# ============================================================
# 전체 재계산 — 결과를 파일로 저장한다
#   화면은 원본(word.csv)을 읽지 않고 이 결과만 읽으므로
#   문서가 늘어나도 화면 속도가 유지된다.
# ============================================================
NOTE_FILE = UNIT_DIR / 'metrics_note.txt'


def recompute_and_save(dict_name=SENTI_DEFAULT, n_cat=SENTI_N_CAT,
                       n_topics=LDA_N_TOPICS):
    """모든 문서의 지표를 다시 계산하고 결과 파일 3종을 저장한다."""
    meta = pd.read_csv(META_CSV, encoding='utf-8-sig')
    if meta.empty:
        pd.DataFrame(columns=METRICS_COLS).to_csv(
            METRICS_CSV,
            index=False,
            encoding="utf-8-sig",
        )

        pd.DataFrame(columns=WORD_FREQ_COLS).to_csv(
            WORD_FREQ_CSV,
            index=False,
            encoding="utf-8-sig",
        )

        pd.DataFrame(columns=DOC_FREQ_COLS).to_csv(
            DOC_FREQ_CSV,
            index=False,
            encoding="utf-8-sig",
        )

        NOTE_FILE.write_text("", encoding="utf-8")

        return pd.DataFrame(columns=METRICS_COLS), None

    word_df = pd.read_csv(UNIT_FILES['word'], encoding='utf-8-sig')
    sent_df = pd.read_csv(UNIT_FILES['sentence'], encoding='utf-8-sig')
    senti, _ = load_sentiment_dict(dict_name)

    # ── 문서별 토큰을 한 번에 모은다 (문서마다 전체를 훑지 않기 위함) ──
    word_df = word_df.dropna(subset=['토큰'])
    word_df['토큰'] = word_df['토큰'].astype(str)
    word_df = word_df[word_df['토큰'].str.strip() != '']
    tok_by_doc = {k: list(v) for k, v in word_df.groupby('파일명')['토큰']}
    sent_by_doc = {k: list(v) for k, v in sent_df.groupby('파일명')['토큰']}

    # ── 내용 다양성 (코퍼스 전체로 LDA 학습) ──
    doc_texts = [(fn, ' '.join(tok_by_doc.get(fn, []))) for fn in meta['파일명']]
    content_div, content_note = compute_content_diversity(doc_texts, n_topics)

    # ── 문서별 지표 ──
    rows = []
    for _, r in meta.iterrows():
        fn = r['파일명']
        toks = tok_by_doc.get(fn, [])
        lens = [sent_token_count(x) for x in sent_by_doc.get(fn, [])]
        lens = [l for l in lens if l > 0]
        sen, matched, rate = compute_sentiment_diversity(toks, senti, n_cat)

        rows.append({
            '파일명': fn, '작성자유형': r['작성자유형'], '문서유형': r['문서유형'],
            S_NSENT: len(lens),
            S_NWORD: len(toks),
            S_NUNIQ: len(set(toks)),
            S_SLEN:  float(np.mean(lens)) if lens else np.nan,
            S_SVAR:  float(np.var(lens)) if len(lens) >= 2 else np.nan,
            D_LEX: compute_lexical_diversity(toks),
            D_CON: content_div.get(fn, np.nan),
            D_SEN: sen,
            D_BUR: compute_burstiness(toks),
            '감성매칭수': matched,
            '감성매칭률': round(rate, 2),
        })
    metrics = pd.DataFrame(rows)
    metrics['감성사전'] = dict_name
    metrics.to_csv(METRICS_CSV, index=False, encoding='utf-8-sig')

    # ── 단어 빈도 집계 (워드클라우드·단어 조회용) ──
    wf = (word_df[~word_df['토큰'].map(is_lone_jamo)]
          .groupby(['작성자유형', '토큰']).size()
          .reset_index(name='빈도')
          .sort_values('빈도', ascending=False))
    wf.to_csv(WORD_FREQ_CSV, index=False, encoding='utf-8-sig')

    df_ = (word_df[~word_df['토큰'].map(is_lone_jamo)]
           .groupby(['파일명', '토큰']).size()
           .reset_index(name='빈도'))
    df_.to_csv(DOC_FREQ_CSV, index=False, encoding='utf-8-sig')

    NOTE_FILE.write_text(content_note or '', encoding='utf-8')
    return metrics, content_note


def metrics_stale():
    """원본이 지표보다 최신이면 재계산이 필요하다"""
    if not METRICS_CSV.exists():
        return True
    m = METRICS_CSV.stat().st_mtime_ns
    for p in [META_CSV, UNIT_FILES['word'], UNIT_FILES['sentence']]:
        if p.exists() and p.stat().st_mtime_ns > m:
            return True
    return False


def file_signature(*paths):
    """파일 변경 여부 서명 (캐시 키)"""
    sig = []
    for p in paths:
        if p.exists():
            s = p.stat()
            sig.append((p.name, s.st_mtime_ns, s.st_size))
    return tuple(sig)


@st.cache_data(show_spinner=False)
def load_metrics(sig):
    if not METRICS_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(METRICS_CSV, encoding='utf-8-sig')
    return df if not df.empty else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_word_freq(sig):
    if not WORD_FREQ_CSV.exists():
        return pd.DataFrame(columns=['작성자유형', '토큰', '빈도'])
    df = pd.read_csv(WORD_FREQ_CSV, encoding='utf-8-sig')
    return df if not df.empty else pd.DataFrame(columns=['작성자유형', '토큰', '빈도'])


@st.cache_data(show_spinner=False)
def load_doc_freq(sig):
    if not DOC_FREQ_CSV.exists():
        return pd.DataFrame(columns=['파일명', '토큰', '빈도'])
    df = pd.read_csv(DOC_FREQ_CSV, encoding='utf-8-sig')
    return df if not df.empty else pd.DataFrame(columns=['파일명', '토큰', '빈도'])


def load_note():
    if NOTE_FILE.exists():
        t = NOTE_FILE.read_text(encoding='utf-8').strip()
        return t or None
    return None


# ============================================================
# 단어 빈도 조회 (집계 파일 기반)
# ============================================================
def word_freq_of(word_freq, atype, stopwords=None, top=None):
    """작성자유형별 단어 빈도 (내림차순)"""
    sub = word_freq[word_freq['작성자유형'] == atype]
    if stopwords:
        sub = sub[~sub['토큰'].isin(stopwords)]
    sub = sub.sort_values('빈도', ascending=False)
    return sub.head(top) if top else sub


def token_stats(word_freq, metrics, token):
    """단어 하나에 대한 작성자유형별 통계"""
    out = {}
    for atype in AUTHOR_TYPES:
        sub = word_freq[word_freq['작성자유형'] == atype]
        total = int(sub['빈도'].sum())
        cnt = int(sub.loc[sub['토큰'] == token, '빈도'].sum())
        ndocs = int((metrics['작성자유형'] == atype).sum()) if not metrics.empty else 0
        out[atype] = {
            'count': cnt, 'total': total,
            'ratio': (cnt / total * 100) if total else 0.0,
            'n_docs': ndocs,
            'per_doc': (cnt / ndocs) if ndocs else 0.0,
        }
    return out


# ============================================================
# 비교 집단
# ============================================================
CASE_BASE_OPTIONS = ['같은 작성자유형', '전체 문서']


def case_base_mask(metrics, cb_choice, target_row):
    if cb_choice == '전체 문서' or target_row is None:
        return pd.Series(True, index=metrics.index)
    return metrics['작성자유형'] == target_row['작성자유형']


def case_base_stats(values):
    v = pd.Series(values).dropna()
    if v.empty:
        return None
    return {'n': int(v.size), 'mean': float(v.mean()),
            'std': float(v.std(ddof=1)) if v.size >= 2 else 0.0,
            'min': float(v.min()), 'max': float(v.max()),
            'values': v.values}


def group_mean(metrics, atype, col):
    v = metrics.loc[metrics['작성자유형'] == atype, col].dropna()
    return float(v.mean()) if not v.empty else np.nan


# ============================================================
# 그래프
#   계산된 지표를 가공하지 않고 실제 수치 그대로 비교한다.
#   ax를 넘기면 그 자리에 그리고, 넘기지 않으면 새 그림을 만든다.
# ============================================================
def _prepare_ax(ax, figsize):
    if ax is not None:
        return ax.figure, ax, False
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax, True


def draw_compare_bars(metric, target_value, metrics, target_name=None,
                      ax=None, figsize=(8.0, 2.8)):
    """
    타겟 문서 · Human 평균 · AI 평균을 실제 수치 그대로 가로 막대로 비교한다.
    """
    fig, ax, standalone = _prepare_ax(ax, figsize)

    labels, values, colors = [], [], []
    if target_name is not None:
        labels.append('현재 문서')
        values.append(target_value)
        colors.append(C_TARGET)
    for atype, color in zip(AUTHOR_TYPES, [C_HUMAN, C_AI]):
        labels.append(f'{atype} 평균')
        values.append(group_mean(metrics, atype, metric))
        colors.append(color)

    labels, values, colors = labels[::-1], values[::-1], colors[::-1]
    plot_vals = [0 if (v is None or np.isnan(v)) else v for v in values]

    bars = ax.barh(labels, plot_vals, color=colors, height=0.55)

    finite = [v for v in values if v is not None and not np.isnan(v)]
    if finite:
        lo, hi = min(finite + [0]), max(finite + [0])
        span = (hi - lo) or (abs(hi) or 1.0)
        ax.set_xlim(lo - span * 0.06, hi + span * 0.26)

    for b, v in zip(bars, values):
        if v is None or np.isnan(v):
            ax.text(0, b.get_y() + b.get_height() / 2, '  계산 없음',
                    va='center', fontsize=11, color='#9AA7B8')
            continue
        off = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.012
        ax.text(v + (off if v >= 0 else -off), b.get_y() + b.get_height() / 2,
                f'{v:.3f}', va='center',
                ha='left' if v >= 0 else 'right',
                fontsize=13, fontweight='bold', color='#1A2332')

    ax.tick_params(labelsize=12)
    ax.set_xlabel('')
    ax.grid(axis='x', color='#EDF1F6', linewidth=1)
    ax.set_axisbelow(True)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color('#D6DEE8')
    if standalone:
        plt.tight_layout()
    return fig


def draw_wordcloud(word_freq, stopwords=None, figsize=(14, 5)):
    """작성자유형별 워드클라우드 (집계된 빈도 사용)"""
    if word_freq.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, atype in zip(axes, AUTHOR_TYPES):
        sub = word_freq_of(word_freq, atype, stopwords)
        if sub.empty:
            ax.set_title(f"{atype} — 데이터 없음", fontsize=13)
            ax.axis('off')
            continue
        freq = dict(zip(sub['토큰'].astype(str), sub['빈도']))
        wc = WordCloud(font_path=KFONT_PATH, width=760, height=520,
                       background_color='white',
                       max_words=100).generate_from_frequencies(freq)
        ax.imshow(wc, interpolation='bilinear')
        ax.set_title(atype, fontsize=15, fontweight='bold')
        ax.axis('off')
    plt.tight_layout()
    return fig


# PDF 보고서 — A4 세로, 화면 내용이 순서대로 이어지도록 구성
# ============================================================
A4_W, A4_H = 8.27, 11.69


# ============================================================
# PDF 전용 색상 체계
# - Human: 차분한 딥 틸
# - AI: 낮은 채도의 테라코타
# - 현재 문서: 해당 작성자 유형의 더 진한 색
# ============================================================
PDF_HUMAN = '#5C7F87'
PDF_HUMAN_DARK = '#294F57'
PDF_HUMAN_LIGHT = '#DDE9EA'
PDF_HUMAN_GROUP_BG = '#F1F6F6'

PDF_AI = '#C2775A'
PDF_AI_DARK = '#914C37'
PDF_AI_LIGHT = '#F2DED7'
PDF_AI_GROUP_BG = '#FBF2EE'

PDF_TEXT = '#1F2937'
PDF_TEXT_SUB = '#667085'
PDF_TEXT_FAINT = '#98A2B3'
PDF_LINE = '#E4E7EC'
PDF_GRID = '#EEF1F4'
PDF_HEADER_BG = '#F2F4F7'
PDF_PANEL_BG = '#FAFAFB'


def pdf_report_doc_colors(author_type):
    """현재 문서 유형에 맞는 진한 색과 옅은 배경색을 반환한다."""
    kind = str(author_type).strip().lower()
    if kind == 'ai':
        return PDF_AI_DARK, PDF_AI_LIGHT
    if kind == 'human':
        return PDF_HUMAN_DARK, PDF_HUMAN_LIGHT
    return '#475467', PDF_HEADER_BG


def pdf_group_colors(group):
    """비교 집단의 기본색, 진한색, 옅은 배경색을 반환한다."""
    if str(group).strip().lower() == 'ai':
        return PDF_AI, PDF_AI_DARK, PDF_AI_GROUP_BG
    return PDF_HUMAN, PDF_HUMAN_DARK, PDF_HUMAN_GROUP_BG


def _new_page(title=None, subtitle=None):
    fig = plt.figure(figsize=(A4_W, A4_H))
    fig.patch.set_facecolor('white')
    if title:
        fig.text(0.08, 0.958, title, fontsize=16, fontweight='bold',
                 va='top', color=PDF_TEXT)
    if subtitle:
        fig.text(0.08, 0.928, subtitle, fontsize=9.5, va='top',
                 color=PDF_TEXT_SUB)
    return fig


def _footer(fig, text):
    fig.text(0.5, 0.028, text, fontsize=8,
             color=PDF_TEXT_FAINT, ha='center')


def _draw_section_title(fig, x, y, text):
    """네 개 주요 섹션 제목을 동일한 대괄호 서식으로 표시한다."""
    fig.text(
        x, y, f'[{text}]',
        fontsize=12.5, fontweight='bold',
        color=PDF_TEXT, va='top'
    )


def _draw_vertical_legend(fig, target_color, author_type):
    """
    기초 통계량 그래프의 3행 2열 빈 공간에
    현재 문서/Human 평균/AI 평균 범례를 세로로 배치한다.
    """

    # 현재 문서의 작성자 유형에 따라 범례 문구를 결정한다.
    author_kind = str(author_type).strip().lower()

    if author_kind == 'human':
        current_label = '현재 문서 (Human 작성)'
    elif author_kind == 'ai':
        current_label = '현재 문서 (AI 작성)'
    else:
        current_label = '현재 문서'

    # 3행 2열의 빈 공간에 범례 축을 배치한다.
    lax = fig.add_axes([
        0.625,   # 왼쪽 위치
        0.452,   # 아래쪽 위치
        0.290,   # 가로 폭
        0.095    # 세로 높이
    ])
    
    lax.set_xlim(0, 1)
    lax.set_ylim(0, 1)
    lax.axis('off')

    # 범례 전체를 감싸는 회색 테두리 박스
    lax.add_patch(
        plt.Rectangle(
            (0.01, 0.02),          # 박스의 왼쪽·아래쪽 위치
            0.98,                  # 박스 가로 길이
            0.96,                  # 박스 세로 길이
            transform=lax.transAxes,
            facecolor='white',     # 박스 내부 배경
            edgecolor='#D6DADF',   # 회색 테두리
            linewidth=0.8,
            clip_on=False,
            zorder=0
        )
    )

    items = [
        (current_label, target_color, 0.78),
        ('Human 평균', PDF_HUMAN, 0.50),
        ('AI 평균', PDF_AI, 0.22),
]

    for text, color, y in items:
        # 막대그래프와 형태를 통일하기 위해 사각형 색상표를 사용한다.
        lax.add_patch(
            plt.Rectangle(
                (0.03, y - 0.07),
                0.07,
                0.14,
                facecolor=color,
                edgecolor='none',
                transform=lax.transAxes,
                clip_on=False
            )
        )

        lax.text(
            0.15,
            y,
            text,
            transform=lax.transAxes,
            va='center',
            ha='left',
            fontsize=8.7,
            color=PDF_TEXT_SUB
        )

def draw_current_doc_top_words(fig, doc_freq, target_name, stopwords,
                               top_n=10, axes_rect=(0.10, 0.06, 0.82, 0.24),
                               bar_color=None):
    """현재 문서에서 가장 많이 쓰인 단어 top_n개를 가로 막대로 그린다."""
    if bar_color is None:
        bar_color = PDF_HUMAN_DARK

    ax = fig.add_axes(axes_rect)
    cur = doc_freq[doc_freq['파일명'] == target_name].copy()
    if not cur.empty and stopwords:
        cur = cur[~cur['토큰'].isin(stopwords)]
    if not cur.empty:
        cur = cur[~cur['토큰'].map(is_lone_jamo)]
    if cur.empty:
        ax.text(0.5, 0.5, '단어 데이터가 없습니다.', ha='center', va='center',
                fontsize=10, color=PDF_TEXT_FAINT)
        ax.axis('off')
        return ax

    top = cur.sort_values('빈도', ascending=False).head(top_n)
    words = list(top['토큰'].astype(str))[::-1]
    freqs = list(top['빈도'])[::-1]

    ax.barh(words, freqs, color=bar_color, height=0.58)
    xmax = max(freqs) or 1
    for i, v in enumerate(freqs):
        ax.text(v + xmax * 0.012, i, f"{int(v)}", va='center',
                fontsize=8.5, color=PDF_TEXT_SUB)

    ax.set_xlim(0, xmax * 1.12)
    ax.tick_params(axis='x', labelsize=8.5, colors=PDF_TEXT_SUB,
                   length=0, pad=5)
    ax.tick_params(axis='y', labelsize=9.5, colors=PDF_TEXT,
                   length=0, pad=8)
    ax.set_xlabel('')
    ax.grid(axis='x', color=PDF_GRID, linewidth=0.8)
    ax.set_axisbelow(True)

    for s in ('top', 'right', 'left', 'bottom'):
        ax.spines[s].set_visible(False)
    return ax


def draw_diversity_mini(ax, metric, tv, metrics, target_color,
                        show_values=True, value_fmt='{:.3f}',
                        value_at_tip=False, y_label_pad=7):
    """PDF용 소형 비교 막대 (현재/Human/AI).
       target_color = 현재 문서 색(작성자유형에 맞춘 진한 색).
       value_at_tip=True면 값을 막대 끝에, False면 오른쪽 한 열로 정렬한다.
       show_values=False면 숫자 라벨을 숨긴다.
    """
    labels = ['현재', 'Human', 'AI']
    vals = [tv, group_mean(metrics, 'Human', metric),
            group_mean(metrics, 'AI', metric)]
    colors = [target_color, PDF_HUMAN, PDF_AI]

    labels, vals, colors = labels[::-1], vals[::-1], colors[::-1]
    plot = [0 if (v is None or np.isnan(v)) else v for v in vals]
    ax.barh(labels, plot, color=colors, height=0.52)

    finite = [v for v in vals if v is not None and not np.isnan(v)]
    if finite:
        lo, hi = min(finite + [0]), max(finite + [0])
        span = (hi - lo) or (abs(hi) or 1.0)
        pad = 0.42 if show_values else 0.08
        ax.set_xlim(lo - span * 0.07, hi + span * pad)

    if show_values:
        xr = ax.get_xlim()
        off = (xr[1] - xr[0]) * 0.015
        for i, v in enumerate(vals):
            if v is None or np.isnan(v):
                ax.text(xr[1], i, '계산 없음', ha='right', va='center',
                        fontsize=8, color=PDF_TEXT_FAINT)
            elif value_at_tip:
                ax.text(v + (off if v >= 0 else -off), i,
                        value_fmt.format(v),
                        ha='left' if v >= 0 else 'right', va='center',
                        fontsize=8.5, fontweight='bold',
                        color=PDF_TEXT, clip_on=False)
            else:
                ax.text(xr[1], i, value_fmt.format(v),
                        ha='right', va='center',
                        fontsize=9, fontweight='bold', color=PDF_TEXT)

    ax.axvline(0, color='#D0D5DD', linewidth=0.7, zorder=0)
    ax.tick_params(axis='x', labelsize=8.0, colors=PDF_TEXT_SUB,
                   length=0, pad=4)
    ax.tick_params(axis='y', labelsize=8.5, colors=PDF_TEXT_SUB,
                   length=0, pad=y_label_pad)
    ax.locator_params(axis='x', nbins=5)
    ax.set_xlabel('')
    ax.grid(axis='x', color=PDF_GRID, linewidth=0.8)
    ax.set_axisbelow(True)

    # 현재 문서 라벨만 굵게 표시
    for label in ax.get_yticklabels():
        if label.get_text() == '현재':
            label.set_fontweight('bold')
            label.set_color(PDF_TEXT)

    for s in ('top', 'right', 'left', 'bottom'):
        ax.spines[s].set_visible(False)
    return ax


def draw_basic_vs_table(fig, trow, metrics, other, axes_rect,
                        hi_bg='#F6EAE5'):
    """현재 문서 vs {other} 집단 표 (other='Human' 또는 'AI').

    열 순서:
    항목 | 현재 문서 | {other} 평균 | vs 평균 | 표준편차 | 최소 | 최대

    수치 계산 방식은 기존과 동일하고, 열 순서와 시각적 강조만 조정한다.
    """
    ax = fig.add_axes(axes_rect)
    ax.axis('off')

    group_color, group_dark, group_light = pdf_group_colors(other)

    headers = [
        '항목', '현재 문서', f'{other} 평균', 'vs 평균',
        '표준편차', '최소', '최대'
    ]
    cells, dirs = [], []

    for c in BASIC_COLS:
        tv = float(trow[c]) if (
            trow is not None and pd.notna(trow[c])
        ) else np.nan
        s = case_base_stats(
            metrics.loc[metrics['작성자유형'] == other, c]
        )

        if s is None:
            cells.append([
                c,
                f"{tv:,.2f}" if not np.isnan(tv) else '-',
                '-', '-', '-', '-', '-'
            ])
            dirs.append(0)
            continue

        gm = s['mean']
        if np.isnan(tv) or abs(gm) < 1e-9:
            ds, d = '-', 0
        else:
            pct = (tv - gm) / gm * 100
            if pct > 0.5:
                ds, d = f"▲ +{pct:.1f}%", 1
            elif pct < -0.5:
                ds, d = f"▼ {pct:.1f}%", -1
            else:
                ds, d = f"≈ {pct:+.1f}%", 0

        dirs.append(d)
        cells.append([
            c,
            f"{tv:,.2f}" if not np.isnan(tv) else '-',
            f"{gm:,.2f}",
            ds,
            f"{s['std']:,.2f}",
            f"{s['min']:,.2f}",
            f"{s['max']:,.2f}",
        ])

    tbl = ax.table(
        cellText=cells,
        colLabels=headers,
        cellLoc='center',
        loc='upper center',
        colWidths=[0.22, 0.15, 0.15, 0.14, 0.13, 0.105, 0.105]
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.52)

    faint = {4, 5, 6}  # 표준편차·최소·최대

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor(PDF_LINE)
        cell.set_linewidth(0.65)
        cell.PAD = 0.055

        if row == 0:
            cell.set_facecolor(PDF_HEADER_BG)
            cell.set_text_props(fontweight='bold', color=PDF_TEXT)
            continue

        # 항목명은 왼쪽 정렬
        if col == 0:
            cell.get_text().set_ha('left')
            cell.set_text_props(color=PDF_TEXT)

        # 현재 문서 열 강조
        if col == 1:
            cell.set_facecolor(hi_bg)
            cell.set_text_props(fontweight='bold', color=PDF_TEXT)

        # 비교 집단 평균 열에 아주 옅은 집단색 적용
        elif col == 2:
            cell.set_facecolor(group_light)
            cell.set_text_props(color=PDF_TEXT)

        # vs 평균을 평균 바로 옆에 배치하고 집단색으로 강조
        elif col == 3:
            cell.set_facecolor(group_light)
            cell.set_text_props(fontweight='bold', color=group_dark)

        # 보조 통계량은 시각적으로 약화
        elif col in faint:
            cell.set_text_props(color=PDF_TEXT_FAINT)

    return ax


def build_pdf_report(metrics, doc_freq, target_name, cb_choice,
                     stopwords, top_n=10):
    """한 문서를 2쪽으로 요약한다.
       1쪽 = 기초 통계량(현재/Human/AI 막대 + 두 비교 표),
       2쪽 = 문서 특성(다양성) + 현재 문서 상위 단어.
    """
    buf = io.BytesIO()
    trow = metrics[metrics['파일명'] == target_name]
    trow = trow.iloc[0] if not trow.empty else None

    n_h = int((metrics['작성자유형'] == 'Human').sum())
    n_a = int((metrics['작성자유형'] == 'AI').sum())

    name_disp = str(target_name or '-')
    if len(name_disp) > 46:
        name_disp = name_disp[:46] + '…'

    a_type = str(trow['작성자유형']) if trow is not None else '-'
    d_type = str(trow['문서유형']) if trow is not None else '-'

    def _tv(c):
        return float(trow[c]) if (
            trow is not None and pd.notna(trow[c])
        ) else np.nan

    # 현재 문서 색 = 작성자유형 색의 진한 버전 (+ 옅은 배경 틴트)
    target_color, target_light = pdf_report_doc_colors(a_type)

    with PdfPages(buf) as pdf:
        # ══ 1쪽 기초 통계량 ══
        fig = _new_page()

        # 헤더: 중립 텍스트 + 현재 문서 유형색 포인트
        fig.add_artist(plt.Line2D(
            [0.06, 0.06], [0.933, 0.965],
            color=target_color, linewidth=4,
            solid_capstyle='round'
        ))
        fig.text(0.075, 0.958, '문서 분석 요약',
                 fontsize=16, fontweight='bold',
                 color=PDF_TEXT, va='top')
        fig.text(0.06, 0.922, name_disp,
                 fontsize=12.5, fontweight='bold',
                 color=PDF_TEXT, va='top')
        fig.text(
            0.06, 0.886, f'{a_type} · {d_type}',
            fontsize=9.3, color=target_color, va='top',
            bbox=dict(
                boxstyle='round,pad=0.32',
                facecolor=target_light,
                edgecolor='none'
            )
        )
        fig.add_artist(plt.Line2D(
            [0.06, 0.94], [0.866, 0.866],
            color=PDF_LINE, linewidth=0.9
        ))

        # ① 현재/Human/AI 3값 막대
        _draw_section_title(
            fig, 0.06, 0.844,
            '기초 통계량 — 현재 문서 vs Human 평균 vs AI 평균'
        )
        fig.text(
            0.06, 0.820,
            f'Human 문서 {n_h}건 · AI 문서 {n_a}건의 평균과 비교',
            fontsize=8.5, color=PDF_TEXT_SUB, va='top'
        )

        # 두 개의 고정 열에 맞춰 배치한다.
        # 마지막 지표도 1열에 정렬하고, 각 그래프의 가로 폭은 축소한다.
        chart_pos = [
            (S_NSENT, (0.115, 0.690, 0.31, 0.060)),
            (S_NWORD, (0.605, 0.690, 0.31, 0.060)),
            (S_SLEN, (0.115, 0.575, 0.31, 0.060)),
            (S_NUNIQ,  (0.605, 0.575, 0.31, 0.060)),
            (S_SVAR,  (0.115, 0.460, 0.31, 0.060)),
        ]

        for c, rect in chart_pos:
            ax = fig.add_axes(rect)
            draw_diversity_mini(
                ax, c, _tv(c), metrics, target_color,
                show_values=True,
                value_fmt='{:,.1f}',
                value_at_tip=True
            )
            ax.set_title(
                c, fontsize=10.3, fontweight='bold',
                color=PDF_TEXT, pad=5
            )

        # 3행 2열의 빈 공간에 세로 범례 배치
        _draw_vertical_legend(
            fig,
            target_color=target_color,
            author_type=a_type
        )

        # ② 두 비교 표 (세로 배치)
        _draw_section_title(
            fig, 0.06, 0.420, '기초 통계량 수치 비교표'
        )


        # 각 표 제목은 표의 가로 중앙에 배치
        fig.text(
            0.50, 0.393, '현재 문서 vs Human 평균',
            fontsize=9.7, fontweight='bold',
            color=PDF_HUMAN_DARK, va='top', ha='center'
        )
        draw_basic_vs_table(
            fig, trow, metrics, 'Human',
            axes_rect=(0.06, 0.244, 0.88, 0.128),
            hi_bg=target_light
        )

        fig.text(
            0.50, 0.214, '현재 문서 vs AI 평균',
            fontsize=9.7, fontweight='bold',
            color=PDF_AI_DARK, va='top', ha='center'
        )
        draw_basic_vs_table(
            fig, trow, metrics, 'AI',
            axes_rect=(0.06, 0.065, 0.88, 0.128),
            hi_bg=target_light
        )

        _footer(fig, '1')
        pdf.savefig(fig)
        plt.close(fig)

        # ══ 2쪽 문서 특성 + 자주 쓰인 단어 ══
        fig = _new_page()

        # ③ 문서 특성
        _draw_section_title(fig, 0.06, 0.955, '문서 특성')

        sub = '현재 문서 · Human 평균 · AI 평균 비교'
        if trow is not None:
            sub += (
                f"      감성 매칭 {int(trow['감성매칭수'])}개 "
                f"({trow['감성매칭률']:.1f}%)"
            )
        fig.text(
            0.06, 0.921, sub,
            fontsize=9.5, color=PDF_TEXT_SUB, va='top'
        )

        positions = [
            (0.10, 0.665, 0.35, 0.16),
            (0.58, 0.665, 0.35, 0.16),
            (0.10, 0.440, 0.35, 0.16),
            (0.58, 0.440, 0.35, 0.16),
        ]

        # 패널을 왼쪽으로 넓혀 '현재/Human/AI' 라벨까지 박스 안에 포함한다.
        for rect in positions:
            x, y, w, h = rect
            fig.add_artist(plt.Rectangle(
                (x - 0.065, y - 0.030),
                w + 0.087, h + 0.070,
                transform=fig.transFigure,
                facecolor=PDF_PANEL_BG,
                edgecolor=PDF_LINE,
                linewidth=0.6,
                zorder=-10
            ))

        for c, rect in zip(DIV_COLS, positions):
            ax = fig.add_axes(rect)
            ax.set_facecolor(PDF_PANEL_BG)
            draw_diversity_mini(
                ax, c, _tv(c), metrics, target_color,
                y_label_pad=4
            )
            ax.set_title(
                c, fontsize=10.8, fontweight='bold',
                color=PDF_TEXT, pad=8
            )

        # ④ 자주 쓰인 단어
        _draw_section_title(
            fig, 0.06, 0.345, f'자주 쓰인 단어 {top_n}개'
        )
        draw_current_doc_top_words(
            fig, doc_freq, target_name, stopwords,
            top_n=top_n,
            axes_rect=(0.10, 0.06, 0.82, 0.225),
            bar_color=target_color
        )

        _footer(fig, '2')
        pdf.savefig(fig)
        plt.close(fig)

    buf.seek(0)
    return buf.getvalue()

# ============================================================
# 드라이브 초기 동기화
# ============================================================
if 'drive_ready' not in st.session_state:
    try:
        with st.spinner("구글 드라이브에서 데이터를 불러오는 중입니다..."):
            svc, fldrs, dtypes = get_folder_structure()
            sync_from_drive(svc, fldrs['output'])
            st.session_state['drive_ready'] = True
            st.session_state['folders']     = fldrs
            st.session_state['doc_types']   = dtypes
    except Exception as e:
        st.error(f"드라이브 연결 실패: {e}")
        st.info("로컬 실행: `.streamlit/secrets.toml` · "
                "클라우드 배포: 앱 Settings → Secrets 설정을 확인해주세요.")
        st.stop()

service       = get_drive_service()
folders       = st.session_state['folders']
DOC_TYPES_ALL = st.session_state['doc_types']
OUTPUT_ID     = folders['output']
ROOT_ID       = folders['root']
init_local_csvs()


# ============================================================
# 사이드바
# ============================================================
with st.sidebar:
    st.header("데이터")

    if st.button("드라이브 스캔", use_container_width=True):
        meta_now = pd.read_csv(META_CSV, encoding='utf-8-sig')
        done = set(meta_now['파일명']) if not meta_now.empty else set()
        pending = []
        with st.spinner("드라이브 폴더를 확인하는 중입니다..."):
            for author in AUTHOR_TYPES:
                for doc in DOC_TYPES_ALL:
                    fid = folders.get((author, doc))
                    if not fid:
                        continue
                    for f in drive_list_pdfs(service, fid):
                        if f['name'] not in done:
                            pending.append({'id': f['id'], 'name': f['name'],
                                            'author': author, 'doc': doc})
        st.session_state['pending'] = pending

    pending = st.session_state.get("pending", [])

    if pending:
        st.success(f"새 문서 {len(pending)}건")

        for p in pending[:6]:
            st.caption(
                f"· [{p['author']}/{p['doc']}] {p['name']}"
            )

        if len(pending) > 6:
            st.caption(f"... 외 {len(pending) - 6}건")

        if st.button(
            "처리 시작",
            type="primary",
            use_container_width=True,
        ):
            prog = st.progress(0.0)
            ok, fail = [], []

            # 1. 새 문서 처리
            for i, p in enumerate(pending):
                try:
                    data = drive_download_bytes(
                        service,
                        p["id"],
                    )

                    counts, err = process_document(
                        p["name"],
                        data,
                        p["doc"],
                        p["author"],
                    )

                    if err:
                        fail.append((p["name"], err))
                    else:
                        ok.append((p["name"], counts))

                except Exception as e:
                    fail.append((p["name"], str(e)))

                finally:
                    prog.progress((i + 1) / len(pending))

            try:
                # 2. 전체 지표 재계산 및 로컬 결과 파일 저장
                with st.spinner("지표를 계산하는 중입니다..."):
                    metrics_updated, content_note = recompute_and_save()

                # 3. CSV 결과 파일을 Drive output 폴더에 동기화
                with st.spinner(
                    "결과 파일을 드라이브에 저장하는 중입니다..."
                ):
                    sync_to_drive(
                        service,
                        OUTPUT_ID,
                    )

                # 4. Google 마스터 시트 자동 갱신
                sheet_id = None
                sheet_error = None

                try:
                    with st.spinner(
                        "마스터 시트를 갱신하는 중입니다..."
                    ):
                        if not has_sheets_permission():
                            raise PermissionError(
                                "구글 시트 권한이 없습니다. "
                                "token.json을 삭제한 뒤 다시 로그인해 주세요."
                            )

                        master_output = metrics_updated.copy()

                        for col in BASIC_COLS:
                            if col in master_output.columns:
                                master_output[col] = (
                                    master_output[col].round(2)
                                )

                        for col in DIV_COLS:
                            if col in master_output.columns:
                                master_output[col] = (
                                    master_output[col].round(4)
                                )

                        sheet_id = ensure_master_sheet(
                            service,
                            OUTPUT_ID,
                        )

                        write_master_sheet(
                            get_sheets_service(),
                            sheet_id,
                            master_output,
                        )

                except Exception as e:
                    sheet_error = str(e)

                # 5. rerun 이후에도 결과 안내를 보여주기 위해 세션에 저장
                st.session_state["process_result"] = {
                    "success_count": len(ok),
                    "failures": fail,
                    "sheet_id": sheet_id,
                    "sheet_error": sheet_error,
                }

                # 처리 목록 초기화
                st.session_state["pending"] = []

                st.cache_data.clear()
                st.rerun()

            except Exception as e:
                st.error(
                    f"지표 계산 또는 결과 파일 저장 중 오류가 발생했습니다: {e}"
                )

                # 문서별 처리 실패 내역 표시
                for name, err in fail:
                    st.error(f"{name}: {err}")


    process_result = st.session_state.pop(
        "process_result",
        None,
        )

    if process_result:
        st.success(
            f"문서 {process_result['success_count']}건의 "
            "분석과 결과 파일 저장을 완료했습니다."
        )

        if process_result["sheet_id"]:
            st.success("마스터 시트를 갱신했습니다.")

            st.markdown(
                f"[📊 마스터 시트 열기]"
                f"({sheet_url(process_result['sheet_id'])})"
            )

        if process_result["sheet_error"]:
            st.warning(
                "분석 결과 파일은 저장되었으나 "
                "마스터 시트 갱신에 실패했습니다: "
                f"{process_result['sheet_error']}"
            )

        for name, err in process_result["failures"]:
            st.error(f"{name}: {err}")

    recompute_result = st.session_state.pop("recompute_result", None)

    if recompute_result:
        st.success(
            "지표 재계산, 결과 파일 저장 및 마스터 시트 갱신을 완료했습니다."
        )
        st.markdown(
            f"[📊 마스터 시트 열기]"
            f"({sheet_url(recompute_result['sheet_id'])})"
        )

    st.divider()
    st.header("화면")

    # ── 상·하위를 하나의 라디오로 합친 계층 메뉴 ──
    TOP_PAGES = ["워드클라우드", "문서 기초 분석", "문서 특성 분석"]
    SUBPAGES = {
        "문서 기초 분석": ["문서별 통계", "단어 조회"],
    }
    SUB_PARENT = {s: p for p, subs in SUBPAGES.items() for s in subs}

    st.session_state.setdefault("menu", "워드클라우드")
    st.session_state.setdefault("sub", None)

    # 현재 활성 화면일 때만 그 하위 항목을 부모 바로 아래에 펼친다 (아코디언)
    nav_options = []
    for page in TOP_PAGES:
        nav_options.append(page)
        if page in SUBPAGES and st.session_state["menu"] == page:
            nav_options.extend(SUBPAGES[page])

    # 하위 항목(문서별 통계·단어 조회)을 상위와 시각적으로 구분한다.
    # 행 전체(동그라미+텍스트)를 들여쓰고 왼쪽에 연결선을 그어 계층을 드러낸다.
    # (하위가 펼쳐졌을 때만 적용: 그때 라디오의 3·4번째 항목이 하위 항목이다.)
    if st.session_state["menu"] in SUBPAGES:
        st.markdown(
            "<style>"
            "section[data-testid='stSidebar'] div[role='radiogroup'] > label:nth-child(3),"
            "section[data-testid='stSidebar'] div[role='radiogroup'] > label:nth-child(4){"
            "margin-left:18px;padding-left:10px;"
            "border-left:2px solid #D8E0EA;font-size:15px;}"
            "</style>",
            unsafe_allow_html=True,
        )

    # 현재 선택 위치 (하위가 활성이면 하위, 아니면 상위 페이지)
    _sel = st.session_state["sub"] or st.session_state["menu"]
    _idx = nav_options.index(_sel) if _sel in nav_options else 0

    choice = st.radio(
        "화면 선택", nav_options, index=_idx,
        label_visibility="collapsed",
    )

    # 사용자가 다른 항목을 클릭하면 상태를 갱신하고 다시 그린다
    # (부모를 누르면 그 화면의 첫 하위 항목이 기본 선택된다)
    if choice != _sel:
        if choice in SUB_PARENT:
            st.session_state["menu"] = SUB_PARENT[choice]
            st.session_state["sub"] = choice
        else:
            st.session_state["menu"] = choice
            st.session_state["sub"] = (
                SUBPAGES[choice][0] if choice in SUBPAGES else None)
        st.rerun()

    menu = st.session_state["menu"]
    sub = st.session_state["sub"]

    st.divider()

    with st.expander("폴더 · 관리"):
        st.markdown(f"[📁 드라이브 폴더 열기]({folder_url(ROOT_ID)})")
        # 마스터시트 열기 — 시트 ID를 한 번만 조회해 세션에 저장(매 렌더마다 API 호출 방지)
        if "master_sheet_id" not in st.session_state:
            st.session_state["master_sheet_id"] = ensure_master_sheet(service, OUTPUT_ID)
        st.markdown(
            f"[📊 마스터시트 열기]({sheet_url(st.session_state['master_sheet_id'])})")
        for author in AUTHOR_TYPES:
            links = [f"[{doc}]({folder_url(folders[(author, doc)])})"
                     for doc in DOC_TYPES_ALL if (author, doc) in folders]
            st.caption(f"**{author}** — " + " · ".join(links))

        st.divider()
        new_type = st.text_input("새 문서 유형 추가", placeholder="예: thesis")
        if st.button("폴더 만들기", use_container_width=True):
            name = new_type.strip()
            if not name:
                st.warning("문서 유형 이름을 입력해 주세요.")
            else:
                for author in AUTHOR_TYPES:
                    drive_ensure_folder(service, folders[author], name)
                st.success(f"'{name}' 폴더를 만들었습니다.")
                del st.session_state['drive_ready']
                st.rerun()

        st.caption("드라이브에서 직접 만든 폴더를 시스템 목록에 반영합니다.")
        if st.button("폴더 구조 새로고침", use_container_width=True):
            if 'drive_ready' in st.session_state:
                del st.session_state['drive_ready']
            st.rerun()

        st.divider()
        if st.button("지표 다시 계산", use_container_width=True):
            try:
                with st.spinner("지표를 다시 계산하는 중입니다..."):
                    metrics_updated, _ = recompute_and_save()

                with st.spinner("결과 파일을 드라이브에 저장하는 중입니다..."):
                    sync_to_drive(service, OUTPUT_ID)

                with st.spinner("마스터 시트를 갱신하는 중입니다..."):
                    if not has_sheets_permission():
                        raise PermissionError(
                            "구글 시트 권한이 없습니다. "
                            "token.json을 삭제한 뒤 다시 로그인해 주세요."
                        )

                    master_output = metrics_updated.copy()

                    for col in BASIC_COLS:
                        if col in master_output.columns:
                            master_output[col] = master_output[col].round(2)

                    for col in DIV_COLS:
                        if col in master_output.columns:
                            master_output[col] = master_output[col].round(4)

                    sheet_id = ensure_master_sheet(service, OUTPUT_ID)

                    write_master_sheet(
                        get_sheets_service(),
                        sheet_id,
                        master_output,
                    )

                st.session_state["recompute_result"] = {
                    "sheet_id": sheet_id
                }

                st.cache_data.clear()
                st.rerun()

            except Exception as e:
                st.error(f"지표 재계산 또는 저장에 실패했습니다: {e}")

        if st.button("전체 초기화", use_container_width=True):
            reset_all(service, OUTPUT_ID)
            st.cache_data.clear()
            st.success("전체 초기화를 완료했습니다.")
            st.rerun()

    if not has_sheets_permission():
        st.warning("구글 시트 저장 권한이 없습니다.\n\n"
                   "`token.json`을 지우고 다시 로그인하세요.\n"
                   "(다른 기능은 정상 작동합니다)")


# ============================================================
# 데이터 로드
# ============================================================
meta = pd.read_csv(META_CSV, encoding='utf-8-sig')

# 원본이 지표보다 최신이면 자동으로 다시 계산한다
if not meta.empty and metrics_stale():
    with st.spinner("지표를 계산하는 중입니다..."):
        recompute_and_save()
    st.cache_data.clear()

metrics   = load_metrics(file_signature(METRICS_CSV))
word_freq = load_word_freq(file_signature(WORD_FREQ_CSV))
content_note = load_note()

custom_sw = load_custom_stopwords()
ALL_SW    = WC_STOPWORDS | custom_sw


# ── 데이터가 없을 때 ──
if meta.empty or metrics.empty:
    st.markdown("<div class='screen-title'>문서 전처리 시스템</div>",
                unsafe_allow_html=True)
    st.markdown("<div class='screen-sub'>아직 분석할 문서가 없습니다.</div>",
                unsafe_allow_html=True)
    st.markdown(f"""
### 시작하는 방법

**1단계** &nbsp; 구글 드라이브 폴더에 PDF 파일을 넣습니다.
&nbsp;&nbsp;&nbsp;&nbsp;[📁 드라이브 폴더 열기]({folder_url(ROOT_ID)})

**2단계** &nbsp; 왼쪽 [드라이브 스캔]을 누릅니다.

**3단계** &nbsp; 찾은 문서가 표시되면 [처리 시작]을 누릅니다.
""")
    st.info("폴더 구성 : Human / AI  →  " + " · ".join(DOC_TYPES_ALL))
    st.stop()


n_human = int((metrics['작성자유형'] == 'Human').sum())
n_ai    = int((metrics['작성자유형'] == 'AI').sum())


# ============================================================
# 화면 1 — 워드클라우드
# ============================================================
if menu == "워드클라우드":
    st.markdown("<div class='screen-title'>워드클라우드</div>",
                unsafe_allow_html=True)
    st.markdown(f"<div class='screen-sub'>"
                f"Human {n_human}건 · AI {n_ai}건</div>",
                unsafe_allow_html=True)

    use_sw = st.toggle("불용어 제거", value=True,
                       help="조사·기능어 등 의미가 적은 단어를 제외합니다.")
    active_sw = ALL_SW if use_sw else None

    fig = draw_wordcloud(word_freq, active_sw)
    if fig is None:
        st.info("표시할 단어가 없습니다.")
    else:
        st.pyplot(fig)

    # ── 불용어 관리 (워드클라우드 바로 아래) ──
    with st.expander(f"불용어 관리  (직접 추가한 단어 {len(custom_sw)}개)"):
        st.caption("워드클라우드에서 빼고 싶은 단어를 등록합니다. "
                   "등록 내용은 드라이브에 저장되어 다음에도 유지됩니다.")
        new_input = st.text_input(
            "추가할 단어",
            placeholder="쉼표 또는 공백으로 구분 (예: 연구, 분석, 결과)")
        if st.button("추가", use_container_width=True):
            added = {w for w in re.split(r'[,\s]+', new_input) if w}
            if added:
                save_custom_stopwords(custom_sw | added)
                st.success(f"{len(added)}개를 추가했습니다.")
                st.rerun()
            else:
                st.warning("추가할 단어를 입력해 주세요.")

        if custom_sw:
            st.divider()
            rm = st.multiselect("삭제할 단어 선택", sorted(custom_sw))
            if st.button("선택 삭제", use_container_width=True):
                if rm:
                    save_custom_stopwords(custom_sw - set(rm))
                    st.success(f"{len(rm)}개를 삭제했습니다.")
                    st.rerun()
                else:
                    st.warning("삭제할 단어를 선택해 주세요.")

    # ── 자주 쓰인 단어: 제목 옆에 표시 개수 선택 ──
    tcol, ncol = st.columns([3, 2])
    with ncol:
        top_n = st.radio("표시할 단어 수", [10, 30], index=0,
                         horizontal=True, key="wc_topn")
    with tcol:
        st.markdown(f"### 자주 쓰인 단어 상위 {top_n}개")

    cols = st.columns(2)
    for col, atype in zip(cols, AUTHOR_TYPES):
        with col:
            st.markdown(f"**{atype}**")
            sub = word_freq_of(word_freq, atype, active_sw, top=top_n)
            if sub.empty:
                st.caption("데이터가 없습니다.")
            else:
                st.dataframe(
                    pd.DataFrame({'순위': range(1, len(sub) + 1),
                                  '단어': sub['토큰'].values,
                                  '빈도': sub['빈도'].values}),
                    use_container_width=True, hide_index=True)


# ============================================================
# 화면 2 — 문서 기초 분석
# ============================================================
elif menu == "문서 기초 분석":
    st.markdown("<div class='screen-title'>문서 기초 분석</div>",
                unsafe_allow_html=True)
    st.markdown(f"<div class='screen-sub'>"
                f"Human {n_human}건 · AI {n_ai}건</div>",
                unsafe_allow_html=True)

    # ── 문서별 통계 ── (사이드바 하위 메뉴로 전환)
    if sub == "문서별 통계":
        st.markdown("### 그룹 평균")
        gm = metrics.groupby('작성자유형')[BASIC_COLS].mean()
        for a in AUTHOR_TYPES:
            if a not in gm.index:
                gm.loc[a] = [np.nan] * len(BASIC_COLS)
        st.dataframe(gm.loc[AUTHOR_TYPES].round(2), use_container_width=True)

        st.markdown("### 문서별 값")
        show = metrics[['파일명', '작성자유형', '문서유형'] + BASIC_COLS].copy()
        for c in [S_SLEN, S_SVAR]:
            show[c] = show[c].round(2)
        st.dataframe(show, use_container_width=True, hide_index=True)

        with st.expander("각 항목의 의미"):
            for c in BASIC_COLS:
                st.markdown(f"**{c}** — {METRIC_DESC.get(c,'')}")

    # ── 단어 조회 ──
    elif sub == "단어 조회":
        st.markdown("### 주요 단어 분포")
        rows = []
        for w in KEY_WORDS:
            s = token_stats(word_freq, metrics, w)
            rows.append({
                '단어': w,
                'Human 빈도': s['Human']['count'],
                'Human 비율(%)': round(s['Human']['ratio'], 3),
                'AI 빈도': s['AI']['count'],
                'AI 비율(%)': round(s['AI']['ratio'], 3),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True,
                     hide_index=True)

        st.divider()
        st.markdown("### 단어 직접 조회")
        token = st.text_input("조회할 단어", key="dist_token",
                              placeholder="여기에 단어를 직접 입력하세요.")

        if token.strip():
            tk = token.strip()
            s = token_stats(word_freq, metrics, tk)
            col_h, col_a = st.columns(2)
            for col, atype in zip([col_h, col_a], AUTHOR_TYPES):
                with col:
                    st.markdown(f"**{atype}** (문서 {s[atype]['n_docs']}건)")
                    st.metric("나온 횟수",    f"{s[atype]['count']:,}회")
                    st.metric("전체 단어 중", f"{s[atype]['ratio']:.2f}%")
                    st.metric("문서당 평균",  f"{s[atype]['per_doc']:.1f}회")

            st.divider()
            st.markdown("#### 문서별로 보기")
            doc_freq = load_doc_freq(file_signature(DOC_FREQ_CSV))
            if doc_freq.empty:
                st.info("문서별 자료가 아직 없습니다. "
                        "왼쪽 [폴더 · 관리]에서 지표를 다시 계산해 주세요.")
            else:
                sel = st.selectbox("문서 선택", list(metrics['파일명']),
                                   key="dist_doc")
                cur = doc_freq[doc_freq['파일명'] == sel]
                total = int(cur['빈도'].sum())
                cnt = int(cur.loc[cur['토큰'] == tk, '빈도'].sum())
                ratio = (cnt / total * 100) if total else 0.0

                m1, m2 = st.columns(2)
                m1.metric("나온 횟수",    f"{cnt}회")
                m2.metric("전체 단어 중", f"{ratio:.2f}%")

                fig, ax = plt.subplots(figsize=(8, 4.5))
                top = cur.sort_values('빈도', ascending=False).head(15)
                colors = [C_TARGET if t == tk else '#B7C2D0'
                          for t in top['토큰']]
                ax.barh(top['토큰'][::-1], top['빈도'][::-1],
                        color=colors[::-1])
                ax.set_title(f"{sel} — 자주 나온 단어 15개", fontsize=12)
                ax.set_xlabel('나온 횟수')
                for s_ in ('top', 'right'):
                    ax.spines[s_].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)
        else:
            st.info("조회할 단어를 입력하면 자세한 분포가 표시됩니다.")


# ============================================================
# 화면 3 — 문서 특성 분석
# ============================================================
elif menu == "문서 특성 분석":
    st.markdown("<div class='screen-title'>문서 특성 분석</div>",
                unsafe_allow_html=True)
    st.markdown(f"<div class='screen-sub'>"
                f"문서가 얼마나 다양한 어휘·주제·감성을 담고 있는지 분석합니다. "
                f"&nbsp;·&nbsp; Human {n_human}건 · AI {n_ai}건</div>",
                unsafe_allow_html=True)

    if content_note:
        st.warning(f"내용 다양성 — {content_note}")

    # ── 문서 선택 (문서가 많아져도 찾을 수 있도록 걸러서 고른다) ──
    st.markdown("### 분석할 문서 선택하기")
    f1, f2, f3 = st.columns([1, 1, 2])
    with f1:
        f_author = st.selectbox("작성자유형", ['전체'] + AUTHOR_TYPES,
                                key="f_author")
    with f2:
        f_doc = st.selectbox("문서유형", ['전체'] + DOC_TYPES_ALL, key="f_doc")
    with f3:
        keyword = st.text_input("파일명 검색", key="f_kw",
                                placeholder="파일명 일부를 입력하세요.")

    cand = metrics.copy()
    if f_author != '전체':
        cand = cand[cand['작성자유형'] == f_author]
    if f_doc != '전체':
        cand = cand[cand['문서유형'] == f_doc]
    if keyword.strip():
        cand = cand[cand['파일명'].astype(str)
                    .str.contains(keyword.strip(), case=False, na=False)]

    if cand.empty:
        st.warning("조건에 맞는 문서가 없습니다. 조건을 바꿔 주세요.")
        st.stop()

    MAX_LIST = 200
    names = list(cand['파일명'])
    if len(names) > MAX_LIST:
        st.caption(f"조건에 맞는 문서 {len(names):,}건 중 {MAX_LIST}건만 "
                   f"목록에 표시합니다. 검색어로 범위를 좁혀 주세요.")
        names = names[:MAX_LIST]

    s1, s2 = st.columns([3, 1])
    with s1:
        target_name = st.selectbox(f"문서 ({len(cand):,}건 중)", names,
                                   key="target_doc")
    with s2:
        cb_choice = st.radio("비교 집단", CASE_BASE_OPTIONS, key="case_base")

    trow  = metrics[metrics['파일명'] == target_name].iloc[0]
    cb_df = metrics[case_base_mask(metrics, cb_choice, trow)]

    if cb_choice == '같은 작성자유형':
        cb_disp = f"같은 작성자유형 {str(trow['작성자유형']).upper()} ({len(cb_df)}건)"
    else:
        cb_disp = f"{cb_choice} ({len(cb_df)}건)"
    st.caption(f"선택한 문서 — **{trow['작성자유형']}** · "
               f"**{trow['문서유형']}**  |  "
               f"비교 집단 — **{cb_disp}**")

    st.divider()

    # ── 지표별 탭 ──
    tabs = st.tabs(DIV_COLS)
    for tab, col in zip(tabs, DIV_COLS):
        with tab:
            st.markdown(
                f"<div class='metric-title'>{col}</div>"
                f"<div class='metric-sub'>{METRIC_EN.get(col, '')}</div>",
                unsafe_allow_html=True
            )

            # 내용 다양성 탭에서만: 표본 부족 경고 (파란 박스 대신 작은 빨간 글씨)
            if col == D_CON and len(metrics) < LDA_MIN_DOCS:
                st.markdown(
                    f"<div style='color:#C0392B; font-size:var(--fs-caption); "
                    f"margin:0 0 var(--space-2) 0;'>"
                    f"⚠️ 현재 수집된 문서가 {len(metrics)}건입니다. "
                    f"내용 다양성은 문서가 {LDA_MIN_DOCS}건 이상일 때 안정적으로 계산됩니다."
                    f"</div>",
                    unsafe_allow_html=True)

            with st.expander("지표의 정의 및 산출식",
                             expanded=False):
                render_diversity_description(col)

            s = case_base_stats(cb_df[col])
            tv = float(trow[col]) if pd.notna(trow[col]) else np.nan

            left, right = st.columns([3, 2])

            with left:
                # 실제 수치를 그대로 비교하는 막대
                st.pyplot(draw_compare_bars(col, tv, metrics, target_name))
                if s:
                    st.markdown(
                        f"<div class='substat'>비교 집단 {s['n']}건 &nbsp;·&nbsp; "
                        f"평균 {s['mean']:.3f} &nbsp;·&nbsp; "
                        f"범위 {s['min']:.3f} ~ {s['max']:.3f} &nbsp;·&nbsp; "
                        f"표준편차 {s['std']:.3f}</div>",
                        unsafe_allow_html=True)

            with right:
                st.metric("현재 문서", "-" if np.isnan(tv) else f"{tv:.3f}")
                if s:
                    st.metric("비교 집단 평균", f"{s['mean']:.3f}")

                if col == D_SEN:
                    st.caption(
                        f"감성사전 {SENTI_DEFAULT} · 긍정·부정 {SENTI_N_CAT}범주 · "
                        f"매칭 {int(trow['감성매칭수'])}개 "
                        f"({trow['감성매칭률']:.1f}%)")
                    if trow['감성매칭률'] < 1:
                        st.warning("매칭된 단어가 매우 적어 값의 신뢰도가 낮습니다.")

            # ── 감성 다양성 탭에만: 다른 사전으로 계산해 보기 (참고 기능) ──
            if col == D_SEN:
                with st.expander("참고 — IBA-DCX 감성사전으로 계산하면"):
                    st.caption("기본 사전은 KNU입니다. "
                               "IBA_DCX는 음식점 리뷰용으로 만들어져 "
                               "학술 문서에는 매칭이 적습니다. "
                               "아래는 비교 참고용입니다.")
                    if st.button("두 사전 비교하기"):
                        with st.spinner("계산 중입니다..."):
                            dfreq = load_doc_freq(file_signature(DOC_FREQ_CSV))
                            by_doc = {k: list(zip(v['토큰'].astype(str), v['빈도']))
                                      for k, v in dfreq.groupby('파일명')}
                            rows = []
                            for dname in SENTI_DICTS:
                                d, derr = load_sentiment_dict(dname)
                                if d is None:
                                    rows.append({'사전': dname, '상태': derr})
                                    continue
                                tot_h, tot_a, rates = [], [], []
                                for _, r in metrics.iterrows():
                                    h, _m, rt = compute_sentiment_diversity_from_freq(
                                        by_doc.get(r['파일명'], []), d)
                                    rates.append(rt)
                                    (tot_h if r['작성자유형'] == 'Human'
                                     else tot_a).append(h)
                                rows.append({
                                    '사전': dname,
                                    '어휘 수': f"{len(d):,}",
                                    'Human 평균': (round(np.nanmean(tot_h), 3)
                                                  if tot_h else np.nan),
                                    'AI 평균': (round(np.nanmean(tot_a), 3)
                                               if tot_a else np.nan),
                                    '평균 매칭률(%)': (round(np.mean(rates), 2)
                                                    if rates else np.nan),
                                })
                        st.dataframe(pd.DataFrame(rows),
                                     use_container_width=True, hide_index=True)

    st.divider()

    with st.expander("전체 문서의 지표 값 보기"):
        show = metrics[['파일명', '작성자유형', '문서유형'] + DIV_COLS].copy()
        for c in DIV_COLS:
            show[c] = show[c].round(4)
        st.dataframe(show, use_container_width=True, hide_index=True)

    # ── 결과 pdf 저장 ──
    st.markdown("### 결과 활용")

    st.caption(
        "현재 분석 결과를 세로형 PDF 보고서로 생성합니다."
    )

    if st.button(
        "보고서(PDF) 만들기",
        use_container_width=True
    ):
        with st.spinner("보고서를 만드는 중입니다..."):
            st.session_state["pdf_bytes"] = build_pdf_report(
                metrics,
                load_doc_freq(file_signature(DOC_FREQ_CSV)),
                target_name,
                cb_choice,
                ALL_SW,
            )

        st.success(
            "보고서를 만들었습니다. 아래에서 내려받으세요."
        )

    if st.session_state.get("pdf_bytes"):
        st.download_button(
            "PDF 내려받기",
            data=st.session_state["pdf_bytes"],
            file_name=(
                f"문서분석보고서_"
                f"{datetime.date.today()}.pdf"
            ),
            mime="application/pdf",
            use_container_width=True,
        )