# -*- coding: utf-8 -*-
"""
Streamlit App：ETF 历史数据 + 平权“韭菜指数”
- 固定三只基金：159915、513050、513120
- 顶部 st.metric 美化：显示韭菜指数的最新指标
- 折线图横坐标为日期，纵坐标缩放并让 1.0 居中，以反映趋势变化，并加 y=1.0 灰色虚线
- 支持基金名称筛选，表格按时间倒序显示
"""

import requests
import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import date
from typing import List

st.set_page_config(page_title="韭菜指数 & 三大ETF")

# ---------------------- 数据抓取与处理 ----------------------
class EastMoneyETFAnalyzer:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        })

    def _market_prefix(self, code: str) -> str:
        return "1" if code.startswith(("6", "5", "9")) else "0"

    def fetch_daily_history(self, fund_code: str, start: str, end: str) -> pd.DataFrame:
        url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
        params = {
            "secid": f"{self._market_prefix(fund_code)}.{fund_code}",
            "fields1": "f1,f2,f3,f4,f5",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
            "klt": "101",
            "fqt": "1",
            "beg": start.replace("-", ""),
            "end": end.replace("-", ""),
        }
        r = self.session.get(url, params=params, timeout=20)
        r.raise_for_status()
        j = r.json()
        klines = (j.get("data") or {}).get("klines")
        name = (j.get("data") or {}).get("name")
        if not klines:
            return pd.DataFrame(columns=["基金代码","基金名称","日期","当日涨跌幅（%）","ETF价格","指数净值"]) 

        rows = []
        base_price = None
        for idx, k in enumerate(klines):
            parts = k.split(",")
            trade_date = parts[0]
            close_price = float(parts[2])
            pct_chg = float(parts[8])
            if idx == 0:
                base_price = close_price
            index_nav = close_price / base_price if base_price else None
            rows.append({
                "基金代码": fund_code,
                "基金名称": name,
                "日期": trade_date,
                "当日涨跌幅（%）": round(pct_chg, 2),
                "ETF价格": close_price,
                "指数净值": round(index_nav, 4) if index_nav else None,
            })
        return pd.DataFrame(rows)

    def fetch_multi_funds(self, fund_codes: List[str], start: str, end: str) -> pd.DataFrame:
        all_data = []
        for code in fund_codes:
            try:
                df = self.fetch_daily_history(code.strip(), start, end)
            except Exception as e:
                st.warning(f"获取 {code} 失败：{e}")
                df = pd.DataFrame()
            if not df.empty:
                all_data.append(df)
        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame(columns=["基金代码","基金名称","日期","当日涨跌幅（%）","ETF价格","指数净值"]) 


def add_equal_weight_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["基金代码","基金名称","日期","当日涨跌幅（%）","ETF价格","指数净值"]) 

    g = (
        df.groupby("日期", as_index=False)["指数净值"].mean()
        .sort_values("日期")
        .reset_index(drop=True)
    )

    nav = g["指数净值"].astype(float).tolist()
    pct_list = [None]
    for i in range(1, len(nav)):
        pct = (nav[i] / nav[i-1] -1) * 100.0
        pct_list.append(round(pct, 2))

    g["基金代码"] = "999999"
    g["基金名称"] = "韭菜指数"
    g["ETF价格"] = None
    g["当日涨跌幅（%）"] = pct_list
    g["指数净值"] = g["指数净值"].round(4)

    cols = ["基金代码","基金名称","日期","当日涨跌幅（%）","ETF价格","指数净值"]
    return g[cols]


@st.cache_data(ttl=600)
def load_data(start: str, end: str) -> pd.DataFrame:
    analyzer = EastMoneyETFAnalyzer()
    fixed_codes = ["159915", "513050", "513120"]
    df = analyzer.fetch_multi_funds(fixed_codes, start, end)
    if df.empty:
        return df
    eq = add_equal_weight_index(df)
    df_all = pd.concat([df, eq], ignore_index=True)
    df_all["日期_dt"] = pd.to_datetime(df_all["日期"])
    df_all = df_all.sort_values(["基金名称", "日期_dt"]).reset_index(drop=True)
    return df_all


# ---------------------- UI ----------------------
st.markdown("## 🥬 韭菜指数追踪")
st.caption("数据来源：东方财富，点击左上方 >> 手动更新")
st.caption("韭指口径：513050/159915/513120等权平均，净值基准日：2025-08-29")
with st.sidebar:
    st.header("时间设置")
    start_d = st.date_input("起始日期", value=date(2025, 8, 29))
    end_d = st.date_input("结束日期", value=date.today())
    fetch_btn = st.button("获取/刷新数据", type="primary")

start_date_str = start_d.strftime("%Y-%m-%d")
end_date_str = end_d.strftime("%Y-%m-%d")

if fetch_btn:
    with st.spinner("抓取数据中，请稍候..."):
        df_plot = load_data(start_date_str, end_date_str)
else:
    df_plot = load_data(start_date_str, end_date_str)

if df_plot.empty:
    st.error("未获取到数据。请检查日期范围是否正确。")
else:
    # 顶部指标：韭菜指数最新
    jiucai_latest = (
        df_plot[df_plot["基金名称"] == "韭菜指数"]
        .sort_values("日期_dt", ascending=False)
        .head(1)
    )
    # 获取最新日期
    latest_date = df_plot["日期_dt"].max().strftime("%Y-%m-%d")
    st.subheader(f"📊 当日韭菜指数指标 （{latest_date}）")

    col1, col2 = st.columns(2)
    if jiucai_latest.empty:
        col1.metric("当日涨跌幅（%）", "—")
        col2.metric("指数净值", "—")
    else:
        v_pct = jiucai_latest.iloc[0]["当日涨跌幅（%）"]
        v_nav = jiucai_latest.iloc[0]["指数净值"]
        col1.metric("当日涨跌幅（%）", "—" if pd.isna(v_pct) else f"{v_pct:.2f}")
        col2.metric("指数净值", f"{v_nav:.4f}")

    # 基金筛选
    all_names = df_plot["基金名称"].dropna().unique().tolist()
    picked = st.multiselect("筛选基金名称", options=all_names, default=all_names)
    df_filtered = df_plot[df_plot["基金名称"].isin(picked)].copy()

    # 折线图：指数净值时间线，plotly绘制，带y=1.0虚线
    st.subheader("📈 韭菜指数净值趋势变化")
    pivot_nav = (
        df_filtered.pivot_table(index="日期_dt", columns="基金名称", values="指数净值")
        .sort_index()
    )
    if not pivot_nav.empty:
        ymin, ymax = pivot_nav.min().min(), pivot_nav.max().max()
        margin = (ymax - ymin) * 0.1
        lower = min(ymin, 1.0) - margin
        upper = max(ymax, 1.0) + margin
        fig = px.line(pivot_nav, x=pivot_nav.index, y=pivot_nav.columns, markers=True)
        fig.update_layout(
            legend=dict(
            orientation="h",  # 水平排列（关键参数，实现平铺）
            yanchor="bottom",
            y=-0.35,  # 位于图表下方
            xanchor="left",
            x=0
            # itemwidth=100,  # 每个图例项宽度，根据需要调整
            # font=dict(size=10)  # 字体大小，避免拥挤
        ),
            yaxis=dict(range=[lower, upper], title="指数净值", zeroline=False),
            xaxis=dict(title="日期"),
            shapes=[dict(type="line", x0=pivot_nav.index.min(), x1=pivot_nav.index.max(),
                         y0=1.0, y1=1.0, line=dict(color="gray", dash="dash"))]
        )
        st.plotly_chart(fig, use_container_width=True)

    # 明细表
    st.subheader("📜 明细数据")
    detail_cols = ["基金代码","基金名称","日期","当日涨跌幅（%）","ETF价格","指数净值"]
    detail_df = (
        df_filtered.sort_values("日期_dt", ascending=False)[detail_cols]
        .reset_index(drop=True)
    )
    st.dataframe(detail_df, use_container_width=True)
