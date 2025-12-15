# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from io import BytesIO

import pandas as pd
import streamlit as st
import xlsxwriter

from analyzer import AnalyzerConfig, analyze_product, extract_asin


def _clean_cell(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return str(value)


def build_report_bytes(df: pd.DataFrame) -> bytes:
    output = BytesIO()

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="分析结果", index=False)

        workbook = writer.book
        worksheet = writer.sheets["分析结果"]

        format_low = workbook.add_format({"bg_color": "#C6EFCE", "font_color": "#006100"})
        format_medium = workbook.add_format({"bg_color": "#FFEB9C", "font_color": "#9C6500"})
        format_high = workbook.add_format({"bg_color": "#FFC7CE", "font_color": "#9C0006"})
        format_fail = workbook.add_format({"bg_color": "#D9D9D9", "font_color": "#000000"})

        header = df.columns.tolist()
        if "综合风险等级" in header:
            risk_col_num = header.index("综合风险等级")
            col_letter = xlsxwriter.utility.xl_col_to_name(risk_col_num)
            last_row = len(df)

            worksheet.conditional_format(
                f"{col_letter}2:{col_letter}{last_row + 1}",
                {"type": "text", "criteria": "containing", "value": "低风险", "format": format_low},
            )
            worksheet.conditional_format(
                f"{col_letter}2:{col_letter}{last_row + 1}",
                {"type": "text", "criteria": "containing", "value": "中风险", "format": format_medium},
            )
            worksheet.conditional_format(
                f"{col_letter}2:{col_letter}{last_row + 1}",
                {"type": "text", "criteria": "containing", "value": "高风险", "format": format_high},
            )
            worksheet.conditional_format(
                f"{col_letter}2:{col_letter}{last_row + 1}",
                {"type": "text", "criteria": "containing", "value": "失败", "format": format_fail},
            )

    return output.getvalue()


st.set_page_config(page_title="选品侵权风险分析", layout="wide")

st.title("选品侵权风险分析")


def _get_secret(name: str) -> str:
    value = os.environ.get(name, "")
    if value:
        return value
    try:
        return str(st.secrets.get(name, ""))
    except Exception:
        return ""


app_password = _get_secret("APP_PASSWORD")
if app_password:
    if "_authed" not in st.session_state:
        st.session_state["_authed"] = False

    if not st.session_state["_authed"]:
        with st.sidebar:
            st.subheader("访问验证")
            pw = st.text_input("访问密码", type="password")
            if st.button("登录"):
                if pw == app_password:
                    st.session_state["_authed"] = True
                else:
                    st.error("密码错误")

        st.info("该应用需要访问密码。请在左侧输入后登录。")
        st.stop()

with st.sidebar:
    st.subheader("配置")

    api_key_secret = _get_secret("GEMINI_API_KEY")
    override_key = st.checkbox("手动输入 API Key（覆盖部署端配置）", value=False)
    if override_key:
        api_key = st.text_input("Gemini API Key", value="", type="password")
    else:
        api_key = api_key_secret
        if api_key_secret:
            st.caption("已使用部署端 Secrets/环境变量中的 GEMINI_API_KEY。")
        else:
            st.caption("未检测到 GEMINI_API_KEY，请手动输入或在部署端配置 Secrets。")

    model = st.text_input("模型", value="gemini-2.5-flash")

    verify_ssl = st.checkbox("启用 SSL 校验", value=True)

    min_delay = st.number_input("请求间最小延迟(秒)", min_value=0.0, max_value=60.0, value=2.0, step=0.5)
    max_delay = st.number_input("请求间最大延迟(秒)", min_value=0.0, max_value=60.0, value=6.0, step=0.5)

    st.caption("提示：亚马逊页面可能会触发验证码，抓取 Brand/Sold By 可能失败。")

st.subheader("上传 Excel")

uploaded = st.file_uploader("选择 Excel 文件 (.xls/.xlsx)", type=["xls", "xlsx"])

st.caption("建议列名：ASIN、产品标题、产品描述、产品图片链接（与你原脚本保持一致）")

if uploaded is None:
    st.stop()

try:
    if uploaded.name.lower().endswith(".xls"):
        df = pd.read_excel(uploaded, engine="xlrd")
    else:
        df = pd.read_excel(uploaded)
except Exception as e:
    st.error(f"读取 Excel 失败：{e}")
    st.stop()

required_columns = ["产品标题", "产品描述"]
missing = [c for c in required_columns if c not in df.columns]
if missing:
    st.error(f"缺少必要列：{', '.join(missing)}")
    st.stop()

if "ASIN" not in df.columns:
    df["ASIN"] = ""
if "产品图片链接" not in df.columns:
    df["产品图片链接"] = ""

result_columns = [
    "Brand (Amazon)",
    "Sold By (Amazon)",
    "综合风险等级",
    "是否符合要求",
    "主要风险类型",
    "风险规避建议",
    "分析理由",
    "侵权溯源链接",
]
for c in result_columns:
    if c not in df.columns:
        df[c] = ""

st.subheader("数据预览")
st.dataframe(df.head(20), use_container_width=True)

run = st.button("开始分析", type="primary", disabled=(not api_key))

if not api_key:
    st.warning("请先在左侧输入 Gemini API Key（或在部署端配置 Secrets/环境变量 GEMINI_API_KEY）。")

if not run:
    st.stop()

cfg = AnalyzerConfig(model=model, min_delay_seconds=float(min_delay), max_delay_seconds=float(max_delay), verify_ssl=bool(verify_ssl))

progress = st.progress(0)
status = st.empty()

total = len(df)

for i in range(total):
    row = df.iloc[i]
    title = _clean_cell(row.get("产品标题", ""))
    desc = _clean_cell(row.get("产品描述", ""))
    asin_raw = _clean_cell(row.get("ASIN", ""))
    image_url = _clean_cell(row.get("产品图片链接", ""))

    asin = extract_asin(asin_raw) or asin_raw

    status.write(f"正在分析 {i + 1}/{total} | ASIN: {asin}")

    res = analyze_product(
        api_key=api_key,
        product_title=title,
        product_desc=desc,
        asin=asin,
        image_url=image_url,
        cfg=cfg,
    )

    for k, v in res.items():
        df.at[df.index[i], k] = _clean_cell(v)

    progress.progress(int(((i + 1) / total) * 100))

st.success("分析完成")

st.subheader("分析结果")
st.dataframe(df, use_container_width=True)

report_bytes = build_report_bytes(df)

st.download_button(
    label="下载 Excel 报表",
    data=report_bytes,
    file_name="选品报表_分析结果.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
