"""
마케팅 테스트 성과 판정기 (A/B/n Test Judge)
실행: streamlit run ab_test_judge.py

입력: 변형(광고/페이지)별 노출 수, 클릭 수, 전환 수
평가: 클릭률(CTR = 클릭/노출), 전환율(CVR = 전환/클릭), 노출 대비 전환율(전환/노출)
검정: 2표본 비율 z-검정 (표본이 작으면 Fisher 정확검정), 다중비교 보정(Bonferroni) 옵션,
      신뢰구간, 상대 개선율(Lift), 베이지안 우위 확률, 필요 표본 수 안내
"""

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from scipy import stats

st.set_page_config(page_title="마케팅 테스트 성과 판정기", page_icon="📊", layout="wide")

# 지표 정의: key -> (표시명, 분자 컬럼, 분모 컬럼, 설명)
METRICS = {
    "ctr": ("클릭률 (CTR)", "클릭수", "노출수", "클릭 ÷ 노출"),
    "cvr": ("전환율 (CVR)", "전환수", "클릭수", "전환 ÷ 클릭"),
    "cpr": ("노출 대비 전환율", "전환수", "노출수", "전환 ÷ 노출 (퍼널 종합)"),
}


# ----------------------------------------------------------------------------
# 통계 함수
# ----------------------------------------------------------------------------
def wilson_ci(x: int, n: int, conf: float):
    """윌슨 신뢰구간 (비율)"""
    if n == 0:
        return 0.0, 0.0
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    p = x / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def compare_proportions(x_c, n_c, x_v, n_v, conf_adj):
    """대조군(c) 대비 변형(v) 비교. p값, 차이, 차이의 신뢰구간, 사용한 검정 반환"""
    p_c, p_v = x_c / n_c, x_v / n_v
    diff = p_v - p_c

    total_x, total_n = x_c + x_v, n_c + n_v
    if total_x == 0 or total_x == total_n:
        p_value, method = 1.0, "검정 불가(변동 없음)"
    else:
        table = np.array([[x_v, n_v - x_v], [x_c, n_c - x_c]])
        expected_min = stats.contingency.expected_freq(table).min()
        if expected_min < 5:
            p_value = stats.fisher_exact(table)[1]
            method = "Fisher 정확검정"
        else:
            p_pool = total_x / total_n
            se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_v))
            z = diff / se
            p_value = 2 * (1 - stats.norm.cdf(abs(z)))
            method = "2표본 비율 z-검정"

    z_crit = stats.norm.ppf(1 - (1 - conf_adj) / 2)
    se_u = np.sqrt(p_v * (1 - p_v) / n_v + p_c * (1 - p_c) / n_c)
    ci = (diff - z_crit * se_u, diff + z_crit * se_u)
    lift = diff / p_c if p_c > 0 else np.nan
    return {"p_c": p_c, "p_v": p_v, "diff": diff, "ci": ci, "lift": lift,
            "p_value": p_value, "method": method}


def bayes_prob_better(x_c, n_c, x_v, n_v, draws=40000, seed=42):
    """Beta(1,1) 사전분포 기반, 변형이 대조군보다 높을 확률"""
    rng = np.random.default_rng(seed)
    s_c = rng.beta(1 + x_c, 1 + n_c - x_c, draws)
    s_v = rng.beta(1 + x_v, 1 + n_v - x_v, draws)
    return float((s_v > s_c).mean())


def required_n(p1, p2, alpha, power=0.8):
    """관측된 효과를 검출하기 위한 그룹당 필요 표본 수 (근사)"""
    if p1 == p2 or p1 <= 0 or p2 <= 0:
        return np.nan
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    n = (z_a + z_b) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2)) / (p1 - p2) ** 2
    return int(np.ceil(n))


def fmt_p(p):
    return "< 0.0001" if p < 0.0001 else f"{p:.4f}"


# ----------------------------------------------------------------------------
# 분석 로직
# ----------------------------------------------------------------------------
def analyze(df, control_name, alpha, bonferroni):
    others = [n for n in df["이름"] if n != control_name]
    m = len(others) if bonferroni else 1
    alpha_adj = alpha / max(m, 1)
    conf_adj = 1 - alpha_adj
    ctrl = df[df["이름"] == control_name].iloc[0]

    out = {}
    for key, (_, num, den, _) in METRICS.items():
        rows = []
        for name in others:
            v = df[df["이름"] == name].iloc[0]
            r = compare_proportions(ctrl[num], ctrl[den], v[num], v[den], conf_adj)
            r["name"] = name
            r["sig"] = r["p_value"] < alpha_adj
            r["bayes"] = bayes_prob_better(ctrl[num], ctrl[den], v[num], v[den])
            r["need_n"] = required_n(r["p_c"], r["p_v"], alpha_adj)
            r["min_n"] = min(ctrl[den], v[den])
            rows.append(r)
        out[key] = rows
    return out, alpha_adj


def decide(rows, control_name):
    """한 지표에 대한 판정 -> (winner 이름 or None, 상태, 메시지)"""
    better = [r for r in rows if r["sig"] and r["diff"] > 0]
    worse = [r for r in rows if r["sig"] and r["diff"] < 0]
    if better:
        best = max(better, key=lambda r: r["p_v"])
        msg = f"**{best['name']}** 이(가) 대조군({control_name}) 대비 유의미하게 우수 (Lift {best['lift']:+.1%})"
        return best["name"], "win", msg
    if worse and len(worse) == len(rows):
        return control_name, "control", f"모든 변형이 대조군보다 유의미하게 낮음 → **{control_name}** 유지"
    if worse:
        names = ", ".join(r["name"] for r in worse)
        return None, "none", f"유의미하게 우수한 변형 없음 ({names}은(는) 대조군보다 유의미하게 낮음)"
    return None, "none", "통계적으로 유의미한 차이 없음"


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.title("📊 마케팅 테스트 성과 판정기")
st.caption("광고/랜딩페이지 테스트 결과를 입력하면 통계적 유의성을 검정해 어떤 안이 더 우수한지 판정합니다.")

with st.sidebar:
    st.header("⚙️ 검정 설정")
    conf_level = st.select_slider("신뢰수준", options=[0.80, 0.90, 0.95, 0.99], value=0.95,
                                  format_func=lambda x: f"{x:.0%}")
    alpha = 1 - conf_level
    primary = st.radio("핵심 평가 지표", list(METRICS.keys()), index=1,
                       format_func=lambda k: METRICS[k][0],
                       help="최종 승자 판정에 사용되는 지표입니다. 나머지는 보조 지표로 함께 점검합니다.")
    bonferroni = st.checkbox("다중비교 보정 (Bonferroni)", value=True,
                             help="변형이 3개 이상일 때 거짓 양성을 줄이기 위해 유의수준을 나눕니다.")
    min_samples = st.number_input("최소 권장 표본(분모 기준)", min_value=0, value=100, step=50,
                                  help="결과를 믿으려면 최소 이 정도는 모여야 한다는 기준 숫자입니다. "
                                       "각 지표의 분모(클릭률·노출 대비 전환율은 노출 수, 전환율은 클릭 수)가 이보다 작으면 "
                                       "신뢰도 경고를 표시합니다. 경고만 띄울 뿐 판정 결과는 바꾸지 않으며, 0으로 두면 경고가 꺼집니다.")
    st.caption("💡 결과를 믿으려면 최소 이 정도는 모여야 한다는 기준입니다. "
               "지표의 분모(노출 수 또는 클릭 수)가 이보다 작으면 경고만 표시하며, 판정 결과는 바뀌지 않습니다.")
    st.divider()
    st.markdown(
        "**지표 정의**\n"
        + "\n".join(f"- {v[0]}: {v[3]}" for v in METRICS.values())
    )
    st.markdown(
        "**전환 수란?**\n\n"
        "광고나 페이지를 본 사람이 **목표로 정한 행동**(구매, 회원가입, 상담 신청, 앱 설치 등)을 "
        "실제로 한 횟수입니다."
    )
    st.caption("💡 A안과 B안에 반드시 같은 기준으로 세어 입력하세요. "
               "(예: 둘 다 '구매 완료'만 집계) 전환 수는 클릭 수보다 클 수 없습니다.")
st.subheader("1. 테스트 데이터 입력")
st.write("표에서 직접 수정하거나 행을 추가하세요. (2개 이상, 첫 번째 행이 기본 대조군)")

default_df = pd.DataFrame({
    "이름": ["A안 (기존)", "B안 (신규)"],
    "노출수": [20000, 20000],
    "클릭수": [600, 700],
    "전환수": [48, 70],
})
edited = st.data_editor(
    default_df, num_rows="dynamic", use_container_width=True, key="data",
    column_config={
        "이름": st.column_config.TextColumn("광고/페이지 이름", required=True),
        "노출수": st.column_config.NumberColumn("노출 수", min_value=0, step=1, format="%d"),
        "클릭수": st.column_config.NumberColumn("클릭 수", min_value=0, step=1, format="%d"),
        "전환수": st.column_config.NumberColumn("전환 수", min_value=0, step=1, format="%d"),
    },
)

# ---- 입력 검증 ----
df = edited.dropna().copy()
errors = []
if len(df) < 2:
    errors.append("비교하려면 최소 2개 이상의 행이 필요합니다.")
else:
    df["이름"] = df["이름"].astype(str).str.strip()
    df["이름"] = df["이름"].replace({"": "변형"})
    for c in ["노출수", "클릭수", "전환수"]:
        df[c] = df[c].astype(int)
    missing_name_mask = df["이름"].str.len() == 0
    if missing_name_mask.any():
        df.loc[missing_name_mask, "이름"] = "변형"
    if df["이름"].duplicated().any():
        seen = {}
        new_names = []
        for name in df["이름"]:
            if name in seen:
                seen[name] += 1
                new_names.append(f"{name} {seen[name]}")
            else:
                seen[name] = 0
                new_names.append(name)
        df["이름"] = new_names
    if (df["노출수"] <= 0).any() or (df["클릭수"] <= 0).any():
        errors.append("노출 수와 클릭 수는 모두 1 이상이어야 합니다. (전환율의 분모가 0이 될 수 없습니다)")
    if (df["클릭수"] > df["노출수"]).any():
        errors.append("클릭 수가 노출 수보다 클 수 없습니다.")
    if (df["전환수"] > df["클릭수"]).any():
        errors.append("전환 수가 클릭 수보다 클 수 없습니다.")

if errors:
    for e in errors:
        st.error(e)
    st.stop()

control_name = st.selectbox("대조군(기준안) 선택", df["이름"].tolist(), index=0)

run_clicked = st.button("🔍 성과 판정하기", type="primary", use_container_width=True)
if not run_clicked:
    st.stop()

results, alpha_adj = analyze(df, control_name, alpha, bonferroni)

# ---- 요약 지표표 ----
st.subheader("2. 성과 한눈에 보기")
summary = df.copy()
for key, (label, num, den, _) in METRICS.items():
    summary[label] = summary[num] / summary[den]
st.dataframe(
    summary.style.format({"노출수": "{:,}", "클릭수": "{:,}", "전환수": "{:,}",
                          **{v[0]: "{:.2%}" for v in METRICS.values()}}),
    use_container_width=True, hide_index=True,
)

# ---- 핵심 지표 판정 ----
st.subheader("3. 핵심 지표 기준 판정")
p_label = METRICS[primary][0]
winner, status, msg = decide(results[primary], control_name)

if status == "win":
    st.success(f"🏆 **핵심 지표({p_label}) 기준 승자: {winner}**\n\n{msg}")
elif status == "control":
    st.warning(f"🛡️ **핵심 지표({p_label}) 기준 대조군 유지**\n\n{msg}")
else:
    st.info(f"⚖️ **핵심 지표({p_label}) 기준 판정 보류**\n\n{msg}\n\n"
            "→ 현재 차이는 방향성은 보이지만, 표본이 충분하지 않아 통계적으로 확실하게 결론을 내릴 수 없습니다. "
            "이 경우는 '승자 없음'이 아니라 '추가 데이터가 필요'로 해석하는 것이 맞습니다.")
    st.caption(
        "💡 해석 팁: 보류는 '실제 차이가 없다'가 아니라, '현재 데이터로는 차이를 확신할 수 없다'는 의미입니다. "
        "특히 전환율은 분모가 클릭 수라서 작은 차이에도 유의성 판정이 어렵습니다."
    )

# 보조 지표 점검
st.markdown("**나머지 지표도 확인 (핵심 지표 승자가 다른 지표에서 손해 보지 않는지)**")
for key in METRICS:
    if key == primary:
        continue
    w, s, m = decide(results[key], control_name)
    icon = {"win": "✅", "control": "🛡️", "none": "➖"}[s]
    st.markdown(f"- {icon} {METRICS[key][0]}: {m}")
    if status == "win" and s == "control":
        st.warning(f"핵심 지표 승자({winner})가 {METRICS[key][0]}에서는 오히려 불리합니다. 지표 간 상충 여부를 확인하세요.")
    elif status == "win":
        for r in results[key]:
            if r["name"] == winner and r["sig"] and r["diff"] < 0:
                st.warning(f"핵심 지표 승자({winner})가 {METRICS[key][0]}에서는 대조군보다 유의미하게 낮습니다. 상충 여부를 확인하세요.")

# ---- 상세 결과 ----
st.subheader("4. 지표별 상세 분석")
st.caption(f"유의수준 α={alpha:.2f}" + (f" → 다중비교 보정 후 α={alpha_adj:.4f}" if bonferroni and alpha_adj != alpha else "")
           + f" · 모든 비교는 대조군({control_name}) 대비")
tabs = st.tabs([METRICS[k][0] for k in METRICS] + ["세 지표 통합"])
for tab, key in zip(tabs, METRICS):
    label, num, den, _ = METRICS[key]
    rows = results[key]
    with tab:
        # 차트: 비율 + 윌슨 신뢰구간
        chart_rows = []
        for _, r in df.iterrows():
            lo, hi = wilson_ci(r[num], r[den], conf_level)
            chart_rows.append({"이름": r["이름"], "비율": r[num] / r[den], "하한": lo, "상한": hi})
        cdf = pd.DataFrame(chart_rows)
        base = alt.Chart(cdf).encode(x=alt.X("이름:N", sort=None, title=None))
        bars = base.mark_bar(opacity=0.75).encode(y=alt.Y("비율:Q", axis=alt.Axis(format="%"), title=label))
        err = base.mark_errorbar().encode(y=alt.Y("하한:Q", title=label), y2="상한:Q")
        st.altair_chart((bars + err).properties(height=280), use_container_width=True)

        table = pd.DataFrame([{
            "변형": r["name"],
            "대조군 비율": r["p_c"],
            "변형 비율": r["p_v"],
            "차이(%p)": r["diff"] * 100,
            "차이 신뢰구간(%p)": f"[{r['ci'][0]*100:+.2f}, {r['ci'][1]*100:+.2f}]",
            "상대 개선율(Lift)": r["lift"],
            "p-value": fmt_p(r["p_value"]),
            "유의성": ("✅ 유의 (우수)" if r["diff"] > 0 else "⚠️ 유의 (열세)") if r["sig"] else "➖ 유의하지 않음",
            "변형이 우수할 확률": r["bayes"],
            "검정 방법": r["method"],
        } for r in rows])
        st.dataframe(
            table.style.format({"대조군 비율": "{:.2%}", "변형 비율": "{:.2%}", "차이(%p)": "{:+.2f}",
                                "상대 개선율(Lift)": "{:+.1%}", "변형이 우수할 확률": "{:.1%}"}),
            use_container_width=True, hide_index=True,
        )

        for r in rows:
            if r["min_n"] < min_samples:
                st.warning(f"{r['name']}: 표본({label} 분모) {r['min_n']:,}건으로 권장 최소치({min_samples:,})보다 적어 신뢰도가 낮습니다.")
            if not r["sig"] and not np.isnan(r["need_n"]):
                st.caption(f"💡 {r['name']}: 현재 관측된 차이가 실제라면 검출력 80%로 유의성을 확인하려면 "
                           f"그룹당 약 **{r['need_n']:,}건**(분모 기준)이 필요합니다. (현재 {r['min_n']:,}건)")

with tabs[-1]:
    ctrl_row = df[df["이름"] == control_name].iloc[0]
    # ---- 통합 그래프: 대조군 = 100 기준 지수 ----
    st.markdown("**세 지표 통합 그래프** (대조군 = 100, 높을수록 좋음)")
    idx_rows = []
    for k, (m_label, m_num, m_den, _) in METRICS.items():
        base_rate = ctrl_row[m_num] / ctrl_row[m_den]
        for _, r in df.iterrows():
            rate = r[m_num] / r[m_den]
            lo, hi = wilson_ci(r[m_num], r[m_den], conf_level)
            idx_rows.append({"지표": m_label, "이름": r["이름"],
                             "지수": rate / base_rate * 100,
                             "하한": lo / base_rate * 100, "상한": hi / base_rate * 100})
    idx_df = pd.DataFrame(idx_rows)
    metric_order = [v[0] for v in METRICS.values()]
    x_enc = alt.X("지표:N", sort=metric_order, title=None, axis=alt.Axis(labelAngle=0))
    bars = alt.Chart(idx_df).mark_bar(opacity=0.8).encode(
        x=x_enc, xOffset=alt.XOffset("이름:N"),
        y=alt.Y("지수:Q", title="대조군 = 100 기준 지수"),
        color=alt.Color("이름:N", title="광고/페이지"),
        tooltip=["지표", "이름", alt.Tooltip("지수:Q", format=".1f")],
    )
    errs = alt.Chart(idx_df).mark_errorbar().encode(
        x=x_enc, xOffset=alt.XOffset("이름:N"),
        y=alt.Y("하한:Q", title="대조군 = 100 기준 지수"), y2="상한:Q",
    )
    ref = alt.Chart(pd.DataFrame({"y": [100]})).mark_rule(strokeDash=[4, 4], color="gray").encode(y="y:Q")
    st.altair_chart((bars + errs + ref).properties(height=340), use_container_width=True)
    st.caption("점선(100)보다 높으면 대조군보다 좋은 지표입니다. 가는 세로선은 신뢰구간이며, 대조군의 선과 많이 겹치면 우연한 차이일 수 있으니 5번 섹션의 🟢/⚪/🔴 판정을 함께 보세요.")

# ============================================================================
# 5. 통합 종합 판정 (CTR + CVR + 노출 대비 전환율)
# ============================================================================
st.subheader("5. 세 지표 종합 판정")
st.caption("세 지표를 가중치로 합산해 어떤 안이 전반적으로 더 나은지 판정합니다.")

wc1, wc2, wc3 = st.columns(3)
w_ctr = wc1.slider("CTR 가중치", 0, 100, 30, 5, key="w_ctr")
w_cvr = wc2.slider("CVR 가중치", 0, 100, 30, 5, key="w_cvr")
w_cpr = wc3.slider("노출 대비 전환율 가중치", 0, 100, 40, 5, key="w_cpr")

if w_ctr + w_cvr + w_cpr == 0:
    st.warning("가중치 합이 0입니다. 하나 이상 0보다 크게 설정하세요.")
else:
    w_sum = w_ctr + w_cvr + w_cpr
    weights = {"ctr": w_ctr / w_sum, "cvr": w_cvr / w_sum, "cpr": w_cpr / w_sum}

    def sample_rates(row, draws, rng):
        """베타 사후분포 샘플. 노출 대비 전환율은 CTR×CVR로 만들어 세 지표가 서로 모순되지 않게 함"""
        ctr = rng.beta(1 + row["클릭수"], 1 + row["노출수"] - row["클릭수"], draws)
        cvr = rng.beta(1 + row["전환수"], 1 + row["클릭수"] - row["전환수"], draws)
        return {"ctr": ctr, "cvr": cvr, "cpr": ctr * cvr}

    rng = np.random.default_rng(7)
    DRAWS = 40000
    ctrl_row = df[df["이름"] == control_name].iloc[0]
    s_c = sample_rates(ctrl_row, DRAWS, rng)
    ICON = {1: "🟢", 0: "⚪", -1: "🔴"}

    integ = []
    for name in [n for n in df["이름"] if n != control_name]:
        v_row = df[df["이름"] == name].iloc[0]
        s_v = sample_rates(v_row, DRAWS, rng)

        # (1) 가중 상대 개선율의 사후분포 -> 통합 우위 확률
        comp = sum(weights[k] * (s_v[k] / s_c[k] - 1) for k in METRICS)
        prob = float((comp > 0).mean())
        med_lift = float(np.median(comp))

        # (2) 지표별 유의성 투표 (유의 우수 +1 / 유의 열세 -1 / 무차이 0), 가중 합산
        signs = {}
        for k in METRICS:
            r = next(x for x in results[k] if x["name"] == name)
            signs[k] = 1 if (r["sig"] and r["diff"] > 0) else (-1 if (r["sig"] and r["diff"] < 0) else 0)
        vote = sum(weights[k] * signs[k] for k in METRICS)

        # (3) 종합 등급
        if vote > 0 and prob >= conf_level:
            code, label = "win", "✅ 통합 우세 (확실)"
        elif prob >= 0.8 and med_lift > 0:
            code, label = "lean_win", "🟡 우세 경향 (추가 데이터 권장)"
        elif vote < 0 and prob <= 1 - conf_level:
            code, label = "lose", "⚠️ 통합 열세 (확실)"
        elif prob <= 0.2:
            code, label = "lean_lose", "🟠 열세 경향"
        else:
            code, label = "tie", "➖ 차이 불분명"

        integ.append({"name": name, "signs": signs, "vote": vote, "prob": prob,
                      "lift": med_lift, "code": code, "label": label})

    # 결과 표
    int_table = pd.DataFrame([{
        "변형": x["name"],
        "CTR": ICON[x["signs"]["ctr"]],
        "CVR": ICON[x["signs"]["cvr"]],
        "노출 대비 전환율": ICON[x["signs"]["cpr"]],
        "유의성 투표 점수": x["vote"],
        "가중 Lift(중앙값)": x["lift"],
        "통합 우위 확률": x["prob"],
        "종합 판정": x["label"],
    } for x in integ])
    st.dataframe(
        int_table.style.format({"유의성 투표 점수": "{:+.2f}", "가중 Lift(중앙값)": "{:+.1%}",
                                "통합 우위 확률": "{:.1%}"}),
        use_container_width=True, hide_index=True,
    )
    st.caption("🟢 대조군보다 유의미하게 우수 · ⚪ 유의미한 차이 없음 · 🔴 유의미하게 열세")

    # 최종 결론
    if integ:
        positives = [x for x in integ if x["code"] in ("win", "lean_win")]
        if positives:
            best = max(positives, key=lambda x: x["prob"])
            text = (f"**{best['name']}** 이(가) 대조군({control_name})보다 전반적으로 더 우수합니다.\n\n"
                    f"가중 개선율 {best['lift']:+.1%} · 통합 우위 확률 {best['prob']:.1%}")
            if best["code"] == "win":
                st.success("🏆 **통합 최종 판정**\n\n" + text)
            else:
                st.info("🟡 **통합 최종 판정 (경향)**\n\n" + text + "\n\n통계적 확실성이 부족하니 데이터를 더 모으는 것을 권장합니다.")
            if any(v == -1 for v in best["signs"].values()):
                bad = [METRICS[k][0] for k, v in best["signs"].items() if v == -1]
                st.warning(f"주의: {best['name']}은(는) {', '.join(bad)}에서 유의미하게 열세입니다. 지표 간 상충을 확인하세요.")
        elif all(x["code"] in ("lose", "lean_lose") for x in integ):
            st.warning(f"🛡️ **통합 최종 판정**: 모든 변형이 대조군보다 열세입니다. **{control_name}** 유지를 권장합니다.")
        else:
            st.info("⚖️ **통합 최종 판정**: 대조군과 변형 사이에 전반적인 우열을 가리기 어렵습니다. 표본을 더 모아 재검정하세요.")
    else:
        st.info("⚖️ **통합 최종 판정**: 비교 대상이 없어 판정을 내릴 수 없습니다. 다른 변형을 추가해 주세요.")

    with st.expander("통합 판정 방식 설명"):
        st.markdown("""
- **유의성 투표 점수**: 지표별 검정 결과(우수 +1 / 열세 −1 / 무차이 0)를 가중 평균한 값. −1 ~ +1 범위입니다.
- **가중 Lift**: 세 지표의 상대 개선율을 가중 평균한 값(사후분포의 중앙값).
- **통합 우위 확률**: 불확실성을 반영해, 가중 개선율이 0보다 클 확률(베이지안 시뮬레이션).
- **✅ 확실**: 투표 점수 > 0 이고 우위 확률이 신뢰수준 이상. **🟡 경향**: 우위 확률 80% 이상.
- 노출 대비 전환율은 CTR×CVR과 같은 값이라 두 지표와 일부 겹칩니다. 중복 반영이 싫다면 해당 가중치를 0으로 두세요.
""")

with st.expander("📘 해석 가이드 / 유의사항"):
    st.markdown("""
- **p-value**: 두 안이 실제로 같다고 가정할 때 지금 같은 차이가 우연히 나올 확률입니다. 보정된 유의수준보다 작으면 '통계적으로 유의'로 판정합니다.
- **차이 신뢰구간**: 0을 포함하지 않으면 유의한 차이이며, 구간 폭이 넓을수록 불확실합니다.
- **변형이 우수할 확률**: 베이지안 관점(균등 사전분포)에서 변형의 실제 성과가 대조군보다 높을 확률입니다. 참고용 지표입니다.
- **표본이 작을 때**: 기대 빈도가 5 미만이면 자동으로 Fisher 정확검정을 사용합니다.
- **최소 권장 표본**: 결과를 믿으려면 최소 이 정도는 모여야 한다는 기준 숫자입니다. 각 지표의 분모(클릭률·노출 대비 전환율은 노출 수, 전환율은 클릭 수)가 이보다 작으면 경고를 띄웁니다. 표본이 적으면 우연의 영향이 커서 결과가 쉽게 뒤집히기 때문입니다. 경고는 판정 결과를 바꾸지 않으며, 데이터를 더 모은 뒤 다시 확인하라는 신호입니다.
- **중간 확인 주의**: 테스트 도중 결과를 반복해서 보고 멈추면 거짓 양성이 늘어납니다. 사전에 정한 기간/표본이 채워진 뒤 판정하세요.
- 통계적 유의성이 곧 비즈니스적 중요성은 아닙니다. 개선 폭(Lift)과 비용 대비 효과도 함께 고려하세요.
""")