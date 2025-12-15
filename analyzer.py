# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, Optional, Tuple

import requests
from bs4 import BeautifulSoup
from google import genai
from google.genai.errors import APIError
from PIL import Image

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9,zh-CN,zh;q=0.7",
}


@dataclass
class AnalyzerConfig:
    model: str = "gemini-2.5-flash"
    timeout_seconds: int = 15
    max_retries: int = 5
    min_delay_seconds: float = 2.0
    max_delay_seconds: float = 6.0
    verify_ssl: bool = True
    headers: Optional[Dict[str, str]] = None


def _sleep_jitter(min_s: float, max_s: float) -> None:
    time.sleep(random.uniform(min_s, max_s))


def extract_asin(value: str) -> Optional[str]:
    if not value:
        return None
    s = str(value).strip()
    m = re.search(r"/dp/([A-Z0-9]{10})", s, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([A-Z0-9]{10})\b", s, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return None


def download_image_as_pil(url: str, cfg: AnalyzerConfig) -> Optional[Image.Image]:
    if not url or not str(url).startswith("http"):
        return None
    try:
        resp = requests.get(
            str(url),
            headers=cfg.headers or DEFAULT_HEADERS,
            timeout=cfg.timeout_seconds,
            verify=cfg.verify_ssl,
        )
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content))
    except Exception:
        return None


def _looks_like_captcha(html: str) -> bool:
    if not html:
        return False
    lower = html.lower()
    return (
        "type the characters" in lower
        or "enter the characters" in lower
        or "automated access" in lower
        or "/errors/validatecaptcha" in lower
        or "captcha" in lower and "amazon" in lower
    )


def get_amazon_seller_info(asin: str, cfg: AnalyzerConfig) -> Tuple[str, str]:
    url = f"https://www.amazon.com/dp/{asin}"
    headers = cfg.headers or DEFAULT_HEADERS

    last_error: Optional[str] = None
    for attempt in range(cfg.max_retries):
        try:
            resp = requests.get(
                url,
                headers=headers,
                timeout=cfg.timeout_seconds,
                verify=cfg.verify_ssl,
            )
            resp.raise_for_status()

            if _looks_like_captcha(resp.text):
                return "Captcha Blocked", "Captcha Blocked"

            soup = BeautifulSoup(resp.text, "lxml")

            brand_element = soup.find("a", id="bylineInfo") or soup.find("span", class_="po-brand")
            brand = brand_element.get_text(strip=True) if brand_element else "Brand Not Found"
            brand = brand.replace("Visit the ", "").replace(" Store", "").strip()

            sold_by_element = soup.find("div", id="merchant-info")
            sold_by = sold_by_element.get_text(" ", strip=True) if sold_by_element else "Sold By Not Found"

            _sleep_jitter(cfg.min_delay_seconds, cfg.max_delay_seconds)
            return brand, sold_by
        except requests.exceptions.RequestException as e:
            last_error = e.__class__.__name__
            if attempt < cfg.max_retries - 1:
                _sleep_jitter(cfg.min_delay_seconds, cfg.max_delay_seconds)
                continue
            return f"爬虫失败 ({last_error})", f"爬虫失败 ({last_error})"
        except Exception as e:
            return f"解析失败: {e}", f"解析失败: {e}"

    return "爬虫失败 (超出重试次数)", "爬虫失败 (超出重试次数)"


def create_analysis_prompt(product_title: str, product_desc: str, brand: str, sold_by: str) -> str:
    return f"""
[角色设定]
你是一位顶级的亚马逊跨境电商选品风险评估专家。你的任务是根据提供的产品信息和图片，对产品进行严格、全面的风险评估。
请务必使用你内置的知识库，对标题、图片和描述中出现的品牌或Logo进行最严格比对。

[待分析的产品文本信息]
产品名称: {product_title}
产品描述/要点: {product_desc}
亚马逊品牌信息: {brand}
亚马逊销售方信息: {sold_by}

[风险禁令清单 - 重点关注商标和擦边球]
请严格检查产品是否触犯以下任一禁令，并在理由中明确指出：
1. 商标/擦边球侵权风险 (高风险 - 重点检查): 标题、描述、产品介绍或图片是否使用了未授权的知名品牌名称或 Logo？
   重要：一旦确认是任何已知大品牌（如 Disney, Nike, Pop Mart, Lego 等）的未授权使用，无论相似度如何，直接判定为"高风险"。
2. 外观/实用专利风险警示：外观设计是否与已知知名品牌设计高度相似？（仅提供高风险警示）
3. 特殊认证/合规风险：是否属于 FDA 认证产品（如食品、医疗器械）、儿童玩具（CPC）或电子产品（涉及 FCC/CE/UL/WEEE/RoHS 等复杂认证）？请在分析理由中明确提示所需合规文件或风险点。
4. 产品形态禁令 (中/高风险)：是否属于粉末、液体、气体、危险品等。

[输出格式要求]
你的输出必须是且仅是一个 JSON 对象。不要包含任何说明文字，不要添加任何前言或总结。
你的输出必须从 {{ 开始，到 }} 结束，且是标准的 JSON 格式。

特别要求：在 "风险规避建议" 字段中，请使用编号列表（1. 2. 3.），并取消使用星号或其他特殊符号。

JSON必须包含以下字段：
{{
  "综合风险等级": "低风险/中风险/高风险",
  "是否符合要求": "是/否",
  "主要风险类型": ["商标侵权", "外观专利", "形态管制", "合规风险", "无"],
  "分析理由": "...",
  "风险规避建议": "..."
}}
""".strip()


def _extract_json(text: str) -> Dict[str, Any]:
    if not text:
        raise json.JSONDecodeError("Empty response", "", 0)

    cleaned = text.strip().replace("```json", "").replace("```", "").strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not m:
            raise json.JSONDecodeError("JSON 匹配失败", cleaned, 0)
        cleaned = m.group(0)
    else:
        cleaned = cleaned[start : end + 1]

    return json.loads(cleaned)


def _build_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


def analyze_product(
    api_key: str,
    product_title: str,
    product_desc: str,
    asin: str,
    image_url: Optional[str],
    cfg: AnalyzerConfig,
    brand_override: Optional[str] = None,
    sold_by_override: Optional[str] = None,
) -> Dict[str, Any]:
    client = _build_client(api_key)

    brand, sold_by = ("待处理", "待处理")
    if brand_override or sold_by_override:
        brand = brand_override or ""
        sold_by = sold_by_override or ""
    else:
        asin_extracted = extract_asin(asin) or asin
        if asin_extracted and len(asin_extracted) == 10:
            brand, sold_by = get_amazon_seller_info(asin_extracted, cfg)

    image = download_image_as_pil(image_url, cfg) if image_url else None

    prompt = create_analysis_prompt(product_title, product_desc, brand, sold_by)
    contents: list[Any] = [prompt]
    if image is not None:
        contents.append(image)

    try:
        resp = client.models.generate_content(model=cfg.model, contents=contents)
        _sleep_jitter(cfg.min_delay_seconds, cfg.max_delay_seconds)
        model_text = getattr(resp, "text", "")
    except APIError as e:
        _sleep_jitter(cfg.min_delay_seconds, cfg.max_delay_seconds)
        return {
            "Brand (Amazon)": brand,
            "Sold By (Amazon)": sold_by,
            "综合风险等级": "API ERROR",
            "是否符合要求": "",
            "主要风险类型": "",
            "风险规避建议": "",
            "分析理由": f"API ERROR: {e}",
            "侵权溯源链接": "",
        }
    except Exception as e:
        _sleep_jitter(cfg.min_delay_seconds, cfg.max_delay_seconds)
        return {
            "Brand (Amazon)": brand,
            "Sold By (Amazon)": sold_by,
            "综合风险等级": "MODEL ERROR",
            "是否符合要求": "",
            "主要风险类型": "",
            "风险规避建议": "",
            "分析理由": f"MODEL ERROR: {e}",
            "侵权溯源链接": "",
        }

    try:
        payload = _extract_json(model_text)
        risk_types = payload.get("主要风险类型", "")
        if isinstance(risk_types, list):
            risk_types_str = ", ".join([str(x) for x in risk_types])
        else:
            risk_types_str = str(risk_types)

        return {
            "Brand (Amazon)": brand,
            "Sold By (Amazon)": sold_by,
            "综合风险等级": payload.get("综合风险等级", "解析失败"),
            "是否符合要求": payload.get("是否符合要求", "解析失败"),
            "主要风险类型": risk_types_str,
            "风险规避建议": payload.get("风险规避建议", ""),
            "分析理由": payload.get("分析理由", model_text),
            "侵权溯源链接": "",
        }
    except Exception as e:
        return {
            "Brand (Amazon)": brand,
            "Sold By (Amazon)": sold_by,
            "综合风险等级": "AI失败/格式错误",
            "是否符合要求": "",
            "主要风险类型": "",
            "风险规避建议": "",
            "分析理由": f"分析失败。错误: {e}. 原始返回: {model_text[:300]}...",
            "侵权溯源链接": "",
        }
