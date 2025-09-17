# streamlit_app.py
"""
Streamlit 앱: 지구온난화 → 대기질(미세먼지) · CO2 연동 대시보드
- 공개 데이터(공식)와 사용자 입력(프롬프트 설명 기반) 대시보드 2개 제공
- 한국어 UI
- 출처(URL)을 아래 주석에 명시
    NOAA Mauna Loa CO2 (월평균): https://gml.noaa.gov/ccgg/trends/data.html
      Raw text: https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.txt
    World Bank PM2.5 indicator: EN.ATM.PM25.MC.M3
      API: http://api.worldbank.org/v2/country/all/indicator/EN.ATM.PM25.MC.M3?format=json&per_page=20000
- 규칙:
    * 전처리 표준화: date, value, group(optional)
    * 미래 데이터 제거(로컬 자정 이후 데이터 삭제)
    * @st.cache_data 사용
    * API 실패 시 재시도 → 실패 시 내부 예시 데이터로 대체하고 사용자에게 안내
    * Pretendard-Bold.ttf 적용 시도 (없으면 자동 생략)
    * 사용자 입력 대시보드는 이 프롬프트의 설명만 사용(파일 업로드 없음)
"""

import io
import sys
import time
from datetime import datetime, timedelta, timezone
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

# ---------------------------
# 설정 & 폰트 시도 (Pretendard)
# ---------------------------
st.set_page_config(page_title="지구온난화·대기질 청소년 건강 대시보드", layout="wide")
LOCAL_TZ = timezone(timedelta(hours=9))  # Asia/Seoul
TODAY_LOCAL = datetime.now(LOCAL_TZ).date()

# Pretendard 적용 시도 (있으면 matplotlib/plotly 텍스트에 사용)
PRETENDARD_PATH = "/fonts/Pretendard-Bold.ttf"
USE_PRETENDARD = False
try:
    font_prop = fm.FontProperties(fname=PRETENDARD_PATH)
    fm.fontManager.addfont(PRETENDARD_PATH)
    plt.rcParams['font.family'] = font_prop.get_name()
    USE_PRETENDARD = True
except Exception:
    # 자동 생략
    USE_PRETENDARD = False

# 표준 헤더
st.markdown("<h1 style='text-align:left'>🌍 지구온난화·대기질 대시보드 — 청소년 기관지 보호 안내</h1>", unsafe_allow_html=True)
st.write("공개 데이터(공식) 기반 대시보드와 사용자 설명 기반 대시보드를 제공합니다. 모든 인터페이스는 한국어로 제공됩니다.")

# ---------------------------
# 유틸리티
# ---------------------------
def remove_future_dates(df, date_col='date'):
    """로컬 자정 이후의 미래 데이터 제거"""
    if date_col not in df.columns:
        return df
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    cutoff = datetime.combine(TODAY_LOCAL, datetime.min.time()).astimezone(LOCAL_TZ)
    return df[df[date_col] <= cutoff]

def safe_request(url, headers=None, params=None, max_retries=2, timeout=15):
    last_exc = None
    for i in range(max_retries):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            last_exc = e
            time.sleep(1 + i)
    raise last_exc

# ---------------------------
# 공개 데이터: NOAA CO2 (Mauna Loa) 불러오기
# 출처: https://gml.noaa.gov/ccgg/trends/data.html
# raw: https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.txt
# ---------------------------
@st.cache_data(ttl=3600)
def load_noaa_co2():
    url = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.txt"
    try:
        r = safe_request(url, max_retries=3)
        text = r.text
        # 파일에는 #으로 시작하는 헤더 라인이 있고 컬럼명이 주석으로 있음.
        lines = [l for l in text.splitlines() if not l.startswith('#') and l.strip()!='']
        # columns per NOAA: year, month, decimal date, average, interpolated, trend, days
        df = pd.read_csv(io.StringIO("\n".join(lines)), sep=r"\s+", header=None,
                         names=['year','month','decimal_date','average','interpolated','trend','days'],
                         engine='python')
        # 월 평균 'interpolated'와 'average'에 -99.99 등 결측 표기 존재
        df['date'] = pd.to_datetime(df['year'].astype(int)*100 + df['month'].astype(int), format='%Y%m')
        # choose interpolated if average is missing
        df['value'] = df['average'].where(df['average']>-900, df['interpolated'])
        df = df[['date','value']].dropna().reset_index(drop=True)
        df = remove_future_dates(df, 'date')
        df['group'] = 'CO2(ppm)'
        return df
    except Exception as e:
        # 예시(대체) 데이터
        dates = pd.date_range(end=TODAY_LOCAL, periods=60, freq='M')
        vals = 410 + np.linspace(-2, 4, len(dates)) + np.random.normal(0,0.2,len(dates))
        df = pd.DataFrame({'date':dates, 'value':vals, 'group':'CO2(ppm)'})
        st.warning("NOAA CO2 데이터 불러오기에 실패하여 예시 데이터로 대체했습니다. (원본: NOAA Mauna Loa) ")
        return df

# ---------------------------
# 공개 데이터: World Bank PM2.5 (indicator EN.ATM.PM25.MC.M3)
# 출처: http://api.worldbank.org/v2/
# ---------------------------
@st.cache_data(ttl=3600)
def load_worldbank_pm25():
    url = "http://api.worldbank.org/v2/country/all/indicator/EN.ATM.PM25.MC.M3"
    params = {"format":"json", "per_page":"20000"}
    try:
        r = safe_request(url, params=params, max_retries=3)
        data = r.json()
        # data[1] contains records
        records = data[1]
        rows = []
        for rec in records:
            country = rec.get('country', {}).get('value')
            year = rec.get('date')
            val = rec.get('value')
            if val is None:
                continue
            try:
                date = pd.to_datetime(f"{year}-01-01")
            except Exception:
                continue
            rows.append({'country':country, 'date':date, 'value':float(val)})
        df = pd.DataFrame(rows)
        # standardize: date, value, group (use country as group)
        df = df.rename(columns={'country':'group'})[['date','value','group']]
        df = remove_future_dates(df, 'date')
        return df
    except Exception as e:
        # 예시(대체) 데이터: 한국(Populated) 연도별 PM2.5
        years = np.arange(TODAY_LOCAL.year-19, TODAY_LOCAL.year+1)
        vals = np.linspace(30, 15, len(years)) + np.random.normal(0,1,len(years))
        df = pd.DataFrame({'date':pd.to_datetime(years.astype(str)+'-01-01'),'value':vals,'group':'Korea'})
        st.warning("World Bank PM2.5 데이터 불러오기에 실패하여 예시 데이터로 대체했습니다. (원본: World Bank PM2.5 indicator) ")
        return df

# ---------------------------
# 데이터 로드
# ---------------------------
with st.spinner("공개 데이터(공식) 불러오는 중..."):
    noaa_df = load_noaa_co2()
    pm25_df = load_worldbank_pm25()

# ---------------------------
# UI: 탭으로 공개/사용자 대시보드 구분
# ---------------------------
tab_public, tab_user = st.tabs(["공개 데이터 대시보드", "사용자 설명 기반 대시보드 (프롬프트)"])

# ---------------------------
# 공개 데이터 대시보드 내용
# ---------------------------
with tab_public:
    st.subheader("공개 데이터: CO₂ (NOAA) · PM2.5 (World Bank)")
    col1, col2 = st.columns([2,1])

    with col1:
        st.markdown("### CO₂ (Mauna Loa) — 월별 자료")
        st.write("출처: NOAA Mauna Loa 시계열. (https://gml.noaa.gov/ccgg/trends/data.html)")

        # 사이드바 옵션 (자동 구성)
        st.sidebar.markdown("## 공개 데이터 옵션")
        # CO2 date range
        co2_min = noaa_df['date'].min().date()
        co2_max = noaa_df['date'].max().date()
        dr_co2 = st.sidebar.date_input("CO₂ 표시 기간", value=(co2_min, co2_max), min_value=co2_min, max_value=co2_max, key='co2_dr')
        smooth_co2 = st.sidebar.slider("CO₂ 스무딩(이동평균 일수)", 0, 24, 6, key='co2_smooth')
        # Main chart
        # filter
        start_co2, end_co2 = pd.to_datetime(dr_co2[0]), pd.to_datetime(dr_co2[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        co2_plot_df = noaa_df[(noaa_df['date']>=start_co2)&(noaa_df['date']<=end_co2)].copy()
        if smooth_co2>0:
            co2_plot_df['value_smooth'] = co2_plot_df['value'].rolling(smooth_co2, min_periods=1).mean()
            y_col = 'value_smooth'
        else:
            y_col = 'value'
        fig_co2 = px.line(co2_plot_df, x='date', y=y_col, labels={'date':'날짜','value':'CO₂ (ppm)','value_smooth':'CO₂ (ppm, 스무딩)'}, title="CO₂ (ppm) 추이")
        fig_co2.update_layout(legend_title_text="")
        st.plotly_chart(fig_co2, use_container_width=True)
        st.download_button("CO₂ 전처리 데이터 다운로드 (CSV)", data=co2_plot_df.to_csv(index=False).encode('utf-8'), file_name='co2_preprocessed.csv')

        st.markdown("---")
        st.markdown("### PM2.5 (World Bank: 연간 평균, 단위 μg/m³)")
        st.write("출처: World Bank indicator EN.ATM.PM25.MC.M3 (http://api.worldbank.org)")

        # PM2.5 options
        groups = pm25_df['group'].unique().tolist()
        sel_country = st.sidebar.multiselect("PM2.5 표시 국가(복수 선택)", options=sorted(groups), default=['Korea'] if 'Korea' in groups else groups[:3], key='pm25_country')
        years = pm25_df['date'].dt.year.sort_values().unique()
        year_min, year_max = int(years.min()), int(years.max())
        yr_range = st.sidebar.slider("PM2.5 연도 범위", year_min, year_max, (max(year_min, year_max-19), year_max), key='yr_range')
        # filter
        pm25_plot = pm25_df[pm25_df['group'].isin(sel_country)].copy()
        pm25_plot = pm25_plot[(pm25_plot['date'].dt.year>=yr_range[0]) & (pm25_plot['date'].dt.year<=yr_range[1])]
        if pm25_plot.empty:
            st.info("선택한 국가/기간에 데이터가 없습니다.")
        else:
            fig_pm = px.line(pm25_plot, x='date', y='value', color='group', labels={'date':'연도','value':'PM2.5 (μg/m³)','group':'국가'}, title="PM2.5 연도별 추이")
            st.plotly_chart(fig_pm, use_container_width=True)
            st.download_button("PM2.5 전처리 데이터 다운로드 (CSV)", data=pm25_plot.to_csv(index=False).encode('utf-8'), file_name='pm25_preprocessed.csv')

    with col2:
        st.markdown("### 데이터 요약")
        st.write("아래는 불러온 공개 데이터의 간단 요약입니다.")
        # summary small tables
        co2_latest = noaa_df.sort_values('date').iloc[-1]
        st.metric("CO₂ 최신 관측 (NOAA, 월평균)", f"{co2_latest['value']:.2f} ppm", delta=None)
        # PM2.5: show selected countries latest
        pm_latest = pm25_df.sort_values('date').groupby('group').tail(1).reset_index(drop=True)
        if not pm_latest.empty:
            st.dataframe(pm_latest.rename(columns={'group':'국가','date':'연도','value':'PM2.5(μg/m³)'}).head(10), height=300)
        st.markdown("---")
        st.markdown("### 데이터 취급 안내")
        st.markdown("- API 호출 실패 시 예시 데이터로 대체됩니다. (화면에 알림 표시)")
        st.markdown("- 오늘(로컬 자정) 이후의 미래 데이터는 자동으로 제거됩니다.")
        st.markdown("- 출처는 코드 주석에 명시되어 있습니다.")

# ---------------------------
# 사용자 설명 기반 대시보드
# Input description: "지구온난화로 인해 대기질의 오염도와 이산화 탄소의 농도를 알아서 청소년의 기관지 건강을 지킬 수 있는 방법을 알려주는 앱이다"
# -> 사용자 데이터는 업로드 없음, 프롬프트 설명만 사용하여 자동 생성된 분석/권장 대시보드 제공
# ---------------------------
with tab_user:
    st.subheader("사용자 설명 기반 대시보드 — 청소년 기관지 보호 안내")
    st.write("입력: 앱 목적 설명(파일 업로드 없음). 이 섹션은 제공된 설명만으로 자동 생성된 데이터·인사이트·권장사항을 제공합니다.")

    # Generate synthetic demo data consistent with description (no external input)
    @st.cache_data(ttl=3600)
    def make_synthetic_user_data():
        # Create monthly CO2 (ppm) and PM2.5 (ug/m3) for a Korean urban area last 36 months
        dates = pd.date_range(end=TODAY_LOCAL, periods=36, freq='M')
        co2 = 410 + np.linspace(-1, 3, len(dates)) + np.random.normal(0,0.3,len(dates))
        pm25 = 25 + np.linspace(5, -3, len(dates)) + 5*np.sin(np.linspace(0,6,len(dates))) + np.random.normal(0,2,len(dates))
        df = pd.DataFrame({'date':dates, 'CO2_ppm':co2, 'PM25_ugm3':pm25})
        df = remove_future_dates(df, 'date')
        return df

    user_df = make_synthetic_user_data()

    # Auto-configure sidebar options for smoothing, risk thresholds
    st.sidebar.markdown("## 사용자 대시보드 옵션 (자동)")
    smooth_window = st.sidebar.slider("그래프 스무딩(이동평균, 개월)", 0, 6, 2, key='user_smooth')
    show_advice = st.sidebar.checkbox("건강권장 표시", value=True, key='show_advice')

    # Visualizations selected automatically
    colA, colB = st.columns(2)
    with colA:
        st.markdown("### (예시) 지역 CO₂ 추이")
        df_plot = user_df.copy()
        if smooth_window>0:
            df_plot['CO2_smooth'] = df_plot['CO2_ppm'].rolling(smooth_window, min_periods=1).mean()
            ycol = 'CO2_smooth'
        else:
            ycol = 'CO2_ppm'
        fig1 = px.area(df_plot, x='date', y=ycol, labels={'date':'날짜', ycol:'CO₂ (ppm)'}, title="지역 CO₂ (예시)")
        st.plotly_chart(fig1, use_container_width=True)

    with colB:
        st.markdown("### (예시) 지역 PM2.5 추이")
        df_plot2 = user_df.copy()
        if smooth_window>0:
            df_plot2['PM25_smooth'] = df_plot2['PM25_ugm3'].rolling(smooth_window, min_periods=1).mean()
            ycol2 = 'PM25_smooth'
        else:
            ycol2 = 'PM25_ugm3'
        fig2 = px.line(df_plot2, x='date', y=ycol2, labels={'date':'날짜', ycol2:'PM2.5 (μg/m³)'}, title="지역 PM2.5 (예시)")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown("### 청소년 기관지 보호 권장 사항 (자동 생성)")
    # Advice logic based on PM2.5 thresholds (WHO/국내 권고 범위)
    # PM2.5 (24-hour): Good <=15, Moderate 15-35, Unhealthy 35-75, Very Unhealthy >75 (illustrative)
    latest_pm = user_df.iloc[-1]['PM25_ugm3']
    latest_co2 = user_df.iloc[-1]['CO2_ppm']
    st.write(f"최근 관측(예시) — PM2.5: **{latest_pm:.1f} μg/m³**, CO₂: **{latest_co2:.1f} ppm**")

    def advice_from_pm(pm):
        if pm <= 15:
            return ("좋음", [
                "야외 활동 제한 불필요 — 통상적인 실외 활동 권장",
                "실내 환기는 평상시대로 진행"
            ])
        elif pm <= 35:
            return ("보통", [
                "민감군(천식, 알레르기 등)은 장시간 격렬한 활동 자제",
                "실외 활동 전 마스크 착용 고려"
            ])
        elif pm <= 75:
            return ("나쁨", [
                "장시간·격렬한 실외 활동 금지",
                "실내 공기질 관리(공기청정기 사용, 창문 닫기 등) 권장",
                "천식·호흡기 질환이 있는 청소년은 의사 상담 권장"
            ])
        else:
            return ("매우 나쁨", [
                "외출 자제, 꼭 외출 시 고성능(편평) 마스크 착용",
                "실내 공기청정기 가동 및 환기 최소화, 의료기관 컨설트"
            ])

    level, recs = advice_from_pm(latest_pm)
    st.markdown(f"**대기질 단계:** {level}")
    for r in recs:
        st.markdown(f"- {r}")

    st.markdown("### CO₂(농도) 관련 안내")
    st.write("CO₂ 농도 자체는 기관지 자극의 직접 지표는 아니지만, 실내 CO₂ 상승은 환기 불량을 의미합니다. 기준(실내): 800 ppm 초과 시 환기 필요 권고.")
    if latest_co2 > 800:
        st.warning("실내 CO₂ 예시값이 800 ppm을 초과했습니다. 실내 환기 및 공기순환을 권장합니다.")
    else:
        st.info("실내 CO₂ 예시값은 비교적 양호합니다.")

    st.markdown("---")
    st.markdown("### 맞춤형 행동 가이드(청소년·학부모용)")
    st.markdown("1. **미세먼지 '나쁨' 이상일 때**: 야외 운동을 실내 운동으로 대체, 마스크 착용(고효율) 권장.\n2. **실내 환기**: CO₂ 기준(≈800 ppm) 이상이면 환기 또는 공기청정기 가동.\n3. **장기 대응**: 주기적 폐 기능 검사(천식 위험군), 학교 공기관리 정책 확인.\n4. **교육**: 청소년 대상 '대기질 읽기' 교육(앱 알림, 학교 방송)을 통해 스스로 행동할 수 있게 함.")
    st.markdown("---")
    st.download_button("사용자(예시) 전처리 데이터 다운로드 (CSV)", data=user_df.to_csv(index=False).encode('utf-8'), file_name='user_demo_preprocessed.csv')

# ---------------------------
# 하단: 개발자 노트 (주석 수준으로 화면에 보이게)
# ---------------------------
st.markdown("---")
st.caption("개발자 노트: 이 앱은 공개 데이터(NOAA, World Bank)와 사용자가 제공한 설명을 바탕으로 자동 생성된 대시보드 예시입니다. 실제 임상·의학적 조치는 전문가 의견을 따르세요.")