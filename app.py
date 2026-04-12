import streamlit as st
import streamlit.components.v1 as components
import pickle
import numpy as np
import pandas as pd
import time
import json
import os
from pathlib import Path
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go

from config import CONFIG
from utils import summarize_dataframe

st.set_page_config(
    page_title="PwC Mule Model Studio",
    page_icon="P",
    layout="wide",
    initial_sidebar_state="expanded"
)

APP_NAME = "Mule Model Studio"
APP_DESCRIPTION = "Synthetic data generation, graph analytics, typology modelling, and alert packaging for mule-risk analysis."
PWC_LOGO_PATH = Path(__file__).resolve().parent / "assets" / "pwc-logo.svg"

PWC_COLORS = {
    "primary": "#D04A02",
    "primary_dark": "#A63B02",
    "primary_light": "#E8773D",
    "secondary": "#2D2D2D",
    "dark": "#1A1A1A",
    "gray_900": "#212121",
    "gray_800": "#333333",
    "gray_700": "#4D4D4D",
    "gray_600": "#666666",
    "gray_500": "#808080",
    "gray_400": "#999999",
    "gray_300": "#B3B3B3",
    "gray_200": "#CCCCCC",
    "gray_100": "#E6E6E6",
    "gray_50": "#F5F5F5",
    "white": "#FFFFFF",
    "success": "#22A861",
    "warning": "#E8A317",
    "danger": "#D32F2F",
    "info": "#2196F3",
    "bg_dark": "#0D0D0D",
    "bg_card": "#1E1E1E",
    "bg_hover": "#2A2A2A",
    "accent_orange": "#D04A02",
    "accent_gold": "#FFB600",
    "accent_teal": "#00A3A1",
    "accent_rose": "#E0457B",
}

UI_THEMES = {
    "Light": {
        "canvas": "#F6F0E8",
        "surface": "#FFFDFC",
        "surface_soft": "#F9F4EE",
        "sidebar": "#241B16",
        "sidebar_soft": "#2F241D",
        "sidebar_text": "#F8EFE6",
        "sidebar_muted": "#D9C5B4",
        "text": "#1E1A17",
        "muted": "#6C645C",
        "line": "#E5D8CB",
        "shadow": "rgba(70, 45, 25, 0.10)",
        "chip": "rgba(208, 74, 2, 0.10)",
        "graph_edge": "rgba(156, 130, 109, 0.35)",
    },
    "Dark": {
        "canvas": "#0F1115",
        "surface": "#171B22",
        "surface_soft": "#1D222B",
        "sidebar": "#0B0E13",
        "sidebar_soft": "#131821",
        "sidebar_text": "#E7ECF4",
        "sidebar_muted": "#B3BDCC",
        "text": "#F3F4F6",
        "muted": "#B7BDC7",
        "line": "#2B3440",
        "shadow": "rgba(0, 0, 0, 0.35)",
        "chip": "rgba(208, 74, 2, 0.16)",
        "graph_edge": "rgba(111, 122, 140, 0.35)",
    },
}

ENTERPRISE_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/icon?family=Material+Icons+Outlined');

:root {{
    --pwc-primary: {PWC_COLORS["primary"]};
    --pwc-dark: {PWC_COLORS["dark"]};
    --pwc-bg: {PWC_COLORS["bg_dark"]};
    --pwc-card: {PWC_COLORS["bg_card"]};
    --pwc-hover: {PWC_COLORS["bg_hover"]};
    --pwc-success: {PWC_COLORS["success"]};
    --pwc-warning: {PWC_COLORS["warning"]};
    --pwc-danger: {PWC_COLORS["danger"]};
}}

.stApp {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: #0A0A0A;
    color: #E0E0E0;
}}

.main .block-container {{
    padding: 1rem 2rem 2rem 2rem;
    max-width: 1500px;
}}

header[data-testid="stHeader"] {{
    background: rgba(10, 10, 10, 0.95);
    backdrop-filter: blur(10px);
}}

div[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #111111 0%, #0D0D0D 100%);
    border-right: 1px solid rgba(208, 74, 2, 0.15);
}}

div[data-testid="stSidebar"] .block-container {{
    padding-top: 0;
}}

div[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
    color: #B0B0B0;
}}

.top-bar {{
    background: linear-gradient(135deg, #111111 0%, #1A1A1A 100%);
    border: 1px solid rgba(208, 74, 2, 0.2);
    border-radius: 16px;
    padding: 1.75rem 2.25rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: relative;
    overflow: hidden;
}}

.top-bar::before {{
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, {PWC_COLORS["primary"]}, {PWC_COLORS["accent_gold"]}, {PWC_COLORS["accent_teal"]});
}}

.top-bar-left h1 {{
    margin: 0;
    font-size: 1.5rem;
    font-weight: 700;
    color: #FFFFFF;
    letter-spacing: -0.5px;
}}

.top-bar-left p {{
    margin: 0.25rem 0 0 0;
    font-size: 0.85rem;
    color: #888888;
    font-weight: 400;
}}

.top-bar-right {{
    display: flex;
    gap: 1.5rem;
    align-items: center;
}}

.top-bar-stat {{
    text-align: center;
}}

.top-bar-stat .value {{
    font-size: 1.3rem;
    font-weight: 700;
    color: {PWC_COLORS["primary"]};
}}

.top-bar-stat .label {{
    font-size: 0.65rem;
    color: #888888;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 0.1rem;
}}

.kpi-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin-bottom: 1.5rem;
}}

.kpi-card {{
    background: #1A1A1A;
    border: 1px solid #2A2A2A;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}}

.kpi-card::after {{
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: var(--accent-color, {PWC_COLORS["primary"]});
    opacity: 0;
    transition: opacity 0.3s ease;
}}

.kpi-card:hover {{
    border-color: rgba(208, 74, 2, 0.3);
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
}}

.kpi-card:hover::after {{
    opacity: 1;
}}

.kpi-card .icon {{
    width: 36px;
    height: 36px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 0.75rem;
    font-size: 1.1rem;
}}

.kpi-card .icon.orange {{ background: rgba(208, 74, 2, 0.15); color: {PWC_COLORS["primary"]}; }}
.kpi-card .icon.green {{ background: rgba(34, 168, 97, 0.15); color: {PWC_COLORS["success"]}; }}
.kpi-card .icon.gold {{ background: rgba(255, 182, 0, 0.15); color: {PWC_COLORS["accent_gold"]}; }}
.kpi-card .icon.teal {{ background: rgba(0, 163, 161, 0.15); color: {PWC_COLORS["accent_teal"]}; }}
.kpi-card .icon.red {{ background: rgba(211, 47, 47, 0.15); color: {PWC_COLORS["danger"]}; }}
.kpi-card .icon.rose {{ background: rgba(224, 69, 123, 0.15); color: {PWC_COLORS["accent_rose"]}; }}

.kpi-card .kpi-label {{
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #888888;
    margin-bottom: 0.25rem;
}}

.kpi-card .kpi-value {{
    font-size: 1.6rem;
    font-weight: 700;
    color: #FFFFFF;
    line-height: 1.2;
}}

.kpi-card .kpi-sub {{
    font-size: 0.72rem;
    color: #666666;
    margin-top: 0.2rem;
}}

.section-title {{
    font-size: 1rem;
    font-weight: 600;
    color: #FFFFFF;
    margin: 1.5rem 0 0.75rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #2A2A2A;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}}

.section-title .material-icons-outlined {{
    font-size: 1.1rem;
    color: {PWC_COLORS["primary"]};
}}

.data-card {{
    background: #1A1A1A;
    border: 1px solid #2A2A2A;
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 1rem;
    transition: all 0.2s ease;
}}

.data-card:hover {{
    border-color: #3A3A3A;
}}

.data-card h3 {{
    margin: 0 0 0.75rem 0;
    font-size: 0.9rem;
    font-weight: 600;
    color: #FFFFFF;
}}

.table-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.6rem 0.75rem;
    border-radius: 8px;
    margin-bottom: 0.25rem;
    transition: background 0.15s ease;
    font-size: 0.82rem;
}}

.table-row:hover {{
    background: #2A2A2A;
}}

.table-row .name {{
    color: #E0E0E0;
    font-weight: 500;
}}

.table-row .info {{
    color: #888888;
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
}}

.badge {{
    display: inline-flex;
    align-items: center;
    padding: 0.2rem 0.65rem;
    border-radius: 20px;
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.3px;
}}

.badge.success {{ background: rgba(34, 168, 97, 0.15); color: {PWC_COLORS["success"]}; }}
.badge.warning {{ background: rgba(232, 163, 23, 0.15); color: {PWC_COLORS["warning"]}; }}
.badge.danger {{ background: rgba(211, 47, 47, 0.15); color: {PWC_COLORS["danger"]}; }}
.badge.info {{ background: rgba(33, 150, 243, 0.15); color: {PWC_COLORS["info"]}; }}
.badge.primary {{ background: rgba(208, 74, 2, 0.15); color: {PWC_COLORS["primary"]}; }}

.progress-steps {{
    display: flex;
    gap: 0.35rem;
    margin-bottom: 1rem;
    padding: 0.75rem 1rem;
    background: #111111;
    border-radius: 10px;
    border: 1px solid #2A2A2A;
}}

.p-step {{
    flex: 1;
    height: 4px;
    border-radius: 2px;
    background: #2A2A2A;
    transition: background 0.3s ease;
}}

.p-step.done {{
    background: {PWC_COLORS["success"]};
}}

.p-step.active {{
    background: {PWC_COLORS["primary"]};
    box-shadow: 0 0 8px rgba(208, 74, 2, 0.4);
}}

.algo-selector {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 0.75rem;
    margin: 0.75rem 0;
}}

.algo-option {{
    background: #1A1A1A;
    border: 2px solid #2A2A2A;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    cursor: pointer;
    transition: all 0.2s ease;
}}

.algo-option:hover {{
    border-color: rgba(208, 74, 2, 0.4);
}}

.algo-option.selected {{
    border-color: {PWC_COLORS["primary"]};
    background: rgba(208, 74, 2, 0.08);
}}

.algo-option h4 {{
    margin: 0;
    font-size: 0.85rem;
    color: #FFFFFF;
    font-weight: 600;
}}

.algo-option p {{
    margin: 0.2rem 0 0 0;
    font-size: 0.7rem;
    color: #888888;
}}

.nav-item {{
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.7rem 1rem;
    border-radius: 10px;
    margin-bottom: 0.25rem;
    cursor: pointer;
    transition: all 0.2s ease;
    text-decoration: none;
    color: #999999;
    font-size: 0.82rem;
    font-weight: 500;
}}

.nav-item:hover {{
    background: rgba(208, 74, 2, 0.08);
    color: #FFFFFF;
}}

.nav-item.active {{
    background: rgba(208, 74, 2, 0.12);
    color: {PWC_COLORS["primary"]};
    font-weight: 600;
}}

.nav-item .nav-icon {{
    width: 28px;
    height: 28px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.85rem;
    flex-shrink: 0;
}}

.nav-item.active .nav-icon {{
    background: rgba(208, 74, 2, 0.15);
}}

.nav-item .nav-number {{
    width: 20px;
    height: 20px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.65rem;
    font-weight: 700;
    margin-left: auto;
}}

.nav-item .nav-number.done {{
    background: rgba(34, 168, 97, 0.15);
    color: {PWC_COLORS["success"]};
}}

.nav-item .nav-number.pending {{
    background: #2A2A2A;
    color: #666666;
}}

.stDataFrame, .stDataEditor {{
    border-radius: 10px;
    overflow: hidden;
}}

div[data-testid="stDataFrame"] > div {{
    border-radius: 10px;
}}

.stTabs [data-baseweb="tab-list"] {{
    gap: 0;
    background: #111111;
    border-radius: 10px;
    padding: 4px;
    border: 1px solid #2A2A2A;
}}

.stTabs [data-baseweb="tab"] {{
    border-radius: 8px;
    font-weight: 500;
    font-size: 0.82rem;
    color: #999999;
    padding: 0.5rem 1rem;
}}

.stTabs [aria-selected="true"] {{
    background: {PWC_COLORS["primary"]};
    color: white;
}}

button[kind="primary"] {{
    background: {PWC_COLORS["primary"]} !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px;
    transition: all 0.2s ease !important;
}}

button[kind="primary"]:hover {{
    background: {PWC_COLORS["primary_dark"]} !important;
    box-shadow: 0 4px 15px rgba(208, 74, 2, 0.3) !important;
}}

button[kind="secondary"] {{
    border-color: #3A3A3A !important;
    color: #E0E0E0 !important;
    border-radius: 8px !important;
}}

.graph-container {{
    background: #111111;
    border: 1px solid #2A2A2A;
    border-radius: 12px;
    overflow: hidden;
    margin: 1rem 0;
}}

.stSelectbox > div > div {{
    background: #1A1A1A;
    border-color: #2A2A2A;
    color: #E0E0E0;
}}

.stNumberInput > div > div > input {{
    background: #1A1A1A;
    border-color: #2A2A2A;
    color: #E0E0E0;
}}

.stSlider > div > div > div {{
    color: {PWC_COLORS["primary"]};
}}

.alert-row {{
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.75rem 1rem;
    border-radius: 8px;
    margin-bottom: 0.35rem;
    border-left: 3px solid;
}}

.alert-row.high {{
    background: rgba(211, 47, 47, 0.08);
    border-left-color: {PWC_COLORS["danger"]};
}}

.alert-row.medium {{
    background: rgba(232, 163, 23, 0.08);
    border-left-color: {PWC_COLORS["warning"]};
}}

.alert-row.low {{
    background: rgba(34, 168, 97, 0.08);
    border-left-color: {PWC_COLORS["success"]};
}}
</style>
"""

st.markdown(ENTERPRISE_CSS, unsafe_allow_html=True)

CLIENT_DEMO_CSS = f"""
<style>
:root {{
    --canvas: #F6F0E8;
    --surface: #FFFDFC;
    --surface-soft: #F9F5F0;
    --ink: #1E1A17;
    --muted: #6C645C;
    --rail: #2A201A;
    --line: #E5D8CB;
}}

.stApp {{
    background:
        radial-gradient(circle at top right, rgba(208, 74, 2, 0.12), transparent 28%),
        linear-gradient(180deg, #FBF6F1 0%, var(--canvas) 100%);
    color: var(--ink);
}}

header[data-testid="stHeader"] {{
    background: rgba(248, 242, 234, 0.88);
    border-bottom: 1px solid rgba(229, 216, 203, 0.9);
}}

.main .block-container {{
    padding: 1rem 2rem 2.25rem 2rem;
    max-width: 1520px;
}}

div[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #2C211B 0%, #201712 100%);
    border-right: 1px solid rgba(255, 255, 255, 0.06);
}}

div[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
div[data-testid="stSidebar"] label {{
    color: #E9DDD3;
}}

.workspace-ribbon {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.55rem;
    margin-bottom: 1rem;
}}

.workspace-chip {{
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    background: rgba(42, 32, 26, 0.95);
    color: #FFF5EF;
    border-radius: 999px;
    padding: 0.45rem 0.9rem;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    box-shadow: 0 10px 24px rgba(42, 32, 26, 0.15);
}}

.workspace-chip.active {{
    background: linear-gradient(135deg, {PWC_COLORS["primary"]}, {PWC_COLORS["primary_light"]});
}}

.top-bar {{
    background: linear-gradient(135deg, rgba(255, 253, 252, 0.98) 0%, rgba(247, 239, 230, 0.94) 100%);
    border: 1px solid var(--line);
    border-radius: 24px;
    padding: 1.9rem 2.1rem;
    box-shadow: 0 18px 40px rgba(70, 45, 25, 0.08);
}}

.top-bar::before {{
    height: 4px;
}}

.top-bar-left h1 {{
    color: var(--ink);
    font-size: 2rem;
}}

.top-bar-left p {{
    color: var(--muted);
    font-size: 0.94rem;
    max-width: 760px;
}}

.top-bar-right {{
    gap: 0.85rem;
    flex-wrap: wrap;
    justify-content: flex-end;
}}

.top-bar-stat {{
    min-width: 112px;
    background: rgba(255, 255, 255, 0.88);
    border: 1px solid rgba(229, 216, 203, 0.95);
    border-radius: 16px;
    padding: 0.75rem 0.9rem;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.55);
}}

.top-bar-stat .value {{
    color: var(--ink);
    font-size: 1.18rem;
}}

.top-bar-stat .label {{
    color: var(--muted);
}}

.kpi-card, .data-card {{
    background: rgba(255, 253, 252, 0.96);
    border: 1px solid var(--line);
    border-radius: 18px;
    box-shadow: 0 12px 28px rgba(70, 45, 25, 0.06);
}}

.kpi-card:hover, .data-card:hover {{
    border-color: rgba(208, 74, 2, 0.28);
    box-shadow: 0 18px 32px rgba(70, 45, 25, 0.10);
}}

.kpi-card .kpi-label, .kpi-card .kpi-sub {{
    color: var(--muted);
}}

.kpi-card .kpi-value, .data-card h3, .section-title {{
    color: var(--ink);
}}

.section-title {{
    border-bottom: 1px solid var(--line);
}}

.progress-steps {{
    background: rgba(255, 253, 252, 0.9);
    border: 1px solid var(--line);
    border-radius: 999px;
    padding: 0.65rem 0.9rem;
}}

.p-step {{
    background: #E6D8CC;
    height: 6px;
}}

.p-step.done {{
    background: #22A861;
}}

.p-step.active {{
    background: {PWC_COLORS["primary"]};
}}

.table-row {{
    background: rgba(255, 255, 255, 0.55);
}}

.table-row:hover {{
    background: rgba(249, 245, 240, 0.95);
}}

.table-row .name {{
    color: var(--ink);
}}

.table-row .info {{
    color: var(--muted);
}}

.stTabs [data-baseweb="tab-list"] {{
    background: rgba(255, 253, 252, 0.98);
    border: 1px solid var(--line);
    padding: 6px;
}}

.stTabs [data-baseweb="tab"] {{
    color: var(--muted);
}}

.stTabs [aria-selected="true"] {{
    color: #FFFFFF;
}}

button[kind="secondary"] {{
    background: rgba(255, 253, 252, 0.94) !important;
    border: 1px solid var(--line) !important;
    color: var(--ink) !important;
}}

.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stDateInput > div > div > input {{
    background: rgba(255, 253, 252, 0.98);
    color: var(--ink);
    border-color: var(--line);
}}

.overview-panel {{
    background: linear-gradient(135deg, rgba(255, 253, 252, 0.96) 0%, rgba(244, 234, 222, 0.92) 100%);
    border: 1px solid var(--line);
    border-radius: 24px;
    padding: 1.6rem 1.75rem;
    box-shadow: 0 18px 36px rgba(70, 45, 25, 0.08);
}}

.overview-panel h3 {{
    margin: 0 0 0.4rem 0;
    font-size: 1.05rem;
    color: var(--ink);
}}

.overview-panel p {{
    margin: 0;
    color: var(--muted);
    font-size: 0.87rem;
}}

.status-pill {{
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.34rem 0.7rem;
    border-radius: 999px;
    font-size: 0.69rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    margin: 0 0.4rem 0.4rem 0;
}}

.status-pill.ready {{
    background: rgba(34, 168, 97, 0.12);
    color: {PWC_COLORS["success"]};
}}

.status-pill.active {{
    background: rgba(208, 74, 2, 0.12);
    color: {PWC_COLORS["primary"]};
}}

.status-pill.pending {{
    background: rgba(42, 32, 26, 0.08);
    color: var(--muted);
}}
</style>
"""

st.markdown(CLIENT_DEMO_CSS, unsafe_allow_html=True)


def current_theme():
    return UI_THEMES.get(st.session_state.get("ui_theme", "Light"), UI_THEMES["Light"])


def inject_theme_css():
    theme = current_theme()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                radial-gradient(circle at top right, rgba(208, 74, 2, 0.10), transparent 24%),
                linear-gradient(180deg, {theme["canvas"]} 0%, {theme["surface_soft"]} 100%) !important;
            color: {theme["text"]} !important;
        }}

        .main .block-container {{
            padding-top: 0.75rem !important;
        }}

        header[data-testid="stHeader"] {{
            background: {theme["surface"]}DD !important;
            border-bottom: 1px solid {theme["line"]} !important;
        }}

        div[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {theme["sidebar"]} 0%, {theme["sidebar_soft"]} 100%) !important;
            border-right: 1px solid {theme["line"]} !important;
        }}

        div[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        div[data-testid="stSidebar"] label,
        div[data-testid="stSidebar"] .stRadio label,
        div[data-testid="stSidebar"] .stCaption {{
            color: {theme["sidebar_muted"]} !important;
        }}

        div[data-testid="stSidebar"] h1,
        div[data-testid="stSidebar"] h2,
        div[data-testid="stSidebar"] h3,
        div[data-testid="stSidebar"] h4,
        div[data-testid="stSidebar"] h5,
        div[data-testid="stSidebar"] h6,
        div[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] strong,
        div[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span {{
            color: {theme["sidebar_text"]} !important;
        }}

        div[data-testid="stSidebar"] .stButton > button {{
            border: 1px solid {theme["line"]} !important;
        }}

        div[data-testid="stSidebar"] button[kind="secondary"] {{
            background: {theme["sidebar_soft"]} !important;
            color: {theme["sidebar_text"]} !important;
        }}

        div[data-testid="stSidebar"] button[kind="primary"] {{
            color: #FFFFFF !important;
        }}

        div[data-testid="stSidebar"] .stSelectbox > div > div,
        div[data-testid="stSidebar"] .stNumberInput > div > div > input,
        div[data-testid="stSidebar"] .stRadio > div {{
            background: {theme["sidebar_soft"]} !important;
            color: {theme["sidebar_text"]} !important;
            border-color: {theme["line"]} !important;
        }}

        .sidebar-brand {{
            font-size: 1.02rem;
            font-weight: 800;
            color: {theme["sidebar_text"]};
        }}

        .sidebar-desc {{
            font-size: 0.76rem;
            line-height: 1.45;
            color: {theme["sidebar_muted"]};
            margin-top: 0.3rem;
        }}

        .sidebar-meta {{
            font-size: 0.72rem;
            color: {theme["sidebar_muted"]};
        }}

        .hero-shell, .overview-panel, .kpi-card, .data-card, .graph-shell, .stat-chip {{
            background: {theme["surface"]} !important;
            color: {theme["text"]} !important;
            border-color: {theme["line"]} !important;
            box-shadow: 0 18px 40px {theme["shadow"]} !important;
        }}

        .hero-shell {{
            border: 1px solid {theme["line"]};
            border-radius: 28px;
            padding: 1.35rem 1.6rem 1.15rem 1.6rem;
            margin-bottom: 0.85rem;
        }}

        .hero-kicker {{
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.38rem 0.75rem;
            border-radius: 999px;
            background: {theme["chip"]};
            color: {PWC_COLORS["primary"]};
            font-size: 0.72rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }}

        .hero-title {{
            margin: 0;
            color: {theme["text"]};
            font-size: 1.85rem;
            line-height: 1.05;
            text-align: left;
        }}

        .hero-subtitle {{
            margin: 0.55rem 0 0 0;
            color: {theme["muted"]};
            font-size: 0.92rem;
            max-width: 900px;
            text-align: left;
        }}

        .hero-stage {{
            margin-top: 0.7rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            flex-wrap: wrap;
        }}

        .hero-stage span {{
            color: {theme["muted"]};
            font-size: 0.84rem;
            font-weight: 600;
        }}

        .hero-progress {{
            flex: 1;
            min-width: 220px;
            height: 8px;
            background: {theme["surface_soft"]};
            border-radius: 999px;
            overflow: hidden;
            border: 1px solid {theme["line"]};
        }}

        .hero-progress > div {{
            height: 100%;
            background: linear-gradient(90deg, {PWC_COLORS["primary"]}, {PWC_COLORS["accent_gold"]});
            border-radius: 999px;
        }}

        .stat-chip {{
            border: 1px solid {theme["line"]};
            border-radius: 18px;
            padding: 0.85rem 0.95rem;
            min-height: 78px;
        }}

        .stat-chip .label {{
            color: {theme["muted"]};
            font-size: 0.74rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 700;
        }}

        .stat-chip .value {{
            color: {theme["text"]};
            font-size: 1.35rem;
            font-weight: 800;
            margin-top: 0.35rem;
        }}

        .stat-chip .sub {{
            color: {theme["muted"]};
            font-size: 0.74rem;
            margin-top: 0.18rem;
        }}

        .stTabs [data-baseweb="tab-list"] {{
            background: {theme["surface"]} !important;
            border: 1px solid {theme["line"]} !important;
        }}

        .stTabs [data-baseweb="tab"] {{
            color: {theme["muted"]} !important;
        }}

        .stTabs [aria-selected="true"] {{
            background: {PWC_COLORS["primary"]} !important;
            color: #FFFFFF !important;
        }}

        button[kind="primary"] {{
            background: linear-gradient(135deg, {PWC_COLORS["primary"]}, {PWC_COLORS["primary_dark"]}) !important;
            color: #FFFFFF !important;
        }}

        button[kind="secondary"] {{
            background: {theme["surface"]} !important;
            border: 1px solid {theme["line"]} !important;
            color: {theme["text"]} !important;
        }}

        .section-title, .kpi-card .kpi-value, .data-card h3, .table-row .name {{
            color: {theme["text"]} !important;
        }}

        .kpi-card .kpi-label, .kpi-card .kpi-sub, .table-row .info {{
            color: {theme["muted"]} !important;
        }}

        .progress-steps, .table-row:hover {{
            background: {theme["surface_soft"]} !important;
            border-color: {theme["line"]} !important;
        }}

        .stSelectbox > div > div,
        .stNumberInput > div > div > input,
        .stDateInput > div > div > input,
        .stTextInput > div > div > input {{
            background: {theme["surface"]} !important;
            color: {theme["text"]} !important;
            border-color: {theme["line"]} !important;
        }}

        .stRadio [role="radiogroup"] label {{
            border-radius: 12px;
            padding: 0.35rem 0.5rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

DARK_PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor="#FFFDFC",
        plot_bgcolor="#FFFDFC",
        font=dict(family="Inter", color="#1E1A17", size=12),
        title_font=dict(size=15, color="#1E1A17"),
        xaxis=dict(gridcolor="#E8DDD2", zerolinecolor="#E8DDD2"),
        yaxis=dict(gridcolor="#E8DDD2", zerolinecolor="#E8DDD2"),
        colorway=[PWC_COLORS["primary"], PWC_COLORS["accent_teal"], PWC_COLORS["accent_gold"],
                  PWC_COLORS["accent_rose"], PWC_COLORS["info"], PWC_COLORS["success"]],
        margin=dict(l=40, r=40, t=50, b=40),
    )
)


def apply_dark_theme(fig):
    theme = current_theme()
    fig.update_layout(
        paper_bgcolor=theme["surface"],
        plot_bgcolor=theme["surface"],
        font=dict(family="Inter", color=theme["text"], size=12),
        title_font=dict(size=15, color=theme["text"]),
        xaxis=dict(gridcolor=theme["line"], zerolinecolor=theme["line"]),
        yaxis=dict(gridcolor=theme["line"], zerolinecolor=theme["line"]),
        colorway=DARK_PLOTLY_TEMPLATE["layout"]["colorway"],
        margin=DARK_PLOTLY_TEMPLATE["layout"]["margin"],
    )
    return fig


def get_pipeline_step_map():
    return [
        ("Data", st.session_state.raw_tables is not None),
        ("Entity", st.session_state.single_view is not None),
        ("Features", st.session_state.feature_df is not None),
        ("Graph", st.session_state.graph_feature_df is not None),
        ("Training", st.session_state.model6_artifacts is not None),
        ("Alerts", st.session_state.alert_output is not None),
        ("Feedback", st.session_state.feedback_outputs is not None),
        ("Export", os.path.exists("best_model.pkl")),
    ]


def render_workspace_ribbon():
    return None


def render_top_bar(title, subtitle, stats=None):
    completed = sum(done for _, done in get_pipeline_step_map())
    progress_pct = completed / 8
    readiness_label = (
        "Pipeline complete"
        if st.session_state.alert_output is not None
        else "Pipeline in progress"
        if st.session_state.feature_df is not None
        else "Ready to run"
    )
    st.markdown(
        f"""
        <div class="hero-shell">
            <div class="hero-kicker">{readiness_label}</div>
            <h1 class="hero-title">{title}</h1>
            <p class="hero-subtitle">{subtitle}</p>
            <div class="hero-stage">
                <span>{completed}/8 stages complete</span>
                <div class="hero-progress"><div style="width:{progress_pct * 100:.0f}%;"></div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if stats:
        stat_cols = st.columns(len(stats))
        for col, stat in zip(stat_cols, stats):
            with col:
                sub = stat.get("sub", "")
                st.markdown(
                    f"""
                    <div class="stat-chip">
                        <div class="label">{stat['label']}</div>
                        <div class="value">{stat['value']}</div>
                        <div class="sub">{sub}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_kpi(icon_class, label, value, sub="", color="orange"):
    st.markdown(f"""
    <div class="kpi-card" style="--accent-color: var(--pwc-primary);">
        <div class="icon {color}">
            <span class="material-icons-outlined">{icon_class}</span>
        </div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


def render_section(title, icon="analytics"):
    st.markdown(f"""
    <div class="section-title">
        <span class="material-icons-outlined">{icon}</span>
        {title}
    </div>
    """, unsafe_allow_html=True)


def render_page_intro(body):
    theme = current_theme()
    st.markdown(
        f"""
        <div style="margin: -0.15rem 0 1.05rem 0; padding: 0.95rem 1.1rem; border-radius: 16px;
                    border: 1px solid {theme["line"]}; background: {theme["surface"]};
                    box-shadow: 0 10px 24px {theme["shadow"]};">
            <div style="font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em;
                        font-weight: 800; color: {PWC_COLORS["primary"]}; margin-bottom: 0.35rem;">
                Stage guide
            </div>
            <div style="font-size: 0.92rem; line-height: 1.65; color: {theme["muted"]};">
                {body}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_note(body):
    theme = current_theme()
    st.markdown(
        f"""
        <div style="margin: 0.25rem 0 0.9rem 0; padding: 0.8rem 1rem; border-radius: 12px;
                    border-left: 4px solid {PWC_COLORS["primary"]}; border-top: 1px solid {theme["line"]};
                    border-right: 1px solid {theme["line"]}; border-bottom: 1px solid {theme["line"]};
                    background: {theme["surface_soft"]};">
            <div style="font-size: 0.88rem; line-height: 1.55; color: {theme["muted"]};">
                {body}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_progress_steps(current, total=8):
    steps_html = ""
    for i in range(1, total + 1):
        cls = "done" if i < current else ("active" if i == current else "")
        steps_html += f'<div class="p-step {cls}"></div>'
    st.markdown(f'<div class="progress-steps">{steps_html}</div>', unsafe_allow_html=True)


def render_network_graph(df, ring_df=None):
    theme = current_theme()
    edges = []
    nodes_set = set()

    sample = df.head(500)
    for _, r in sample.iterrows():
        cust = str(r.get("customer_id", ""))
        acct = str(r.get("account_id", ""))
        cp = str(r.get("counterparty_id", ""))

        if cust and acct and cust != "nan" and acct != "nan":
            nodes_set.add(cust)
            nodes_set.add(acct)
            edges.append({"from": cust, "to": acct, "type": "owns"})

        if acct and cp and acct != "nan" and cp != "nan":
            nodes_set.add(acct)
            nodes_set.add(cp)
            edges.append({"from": acct, "to": cp, "type": "transfers"})

    ring_paths = []
    if ring_df is not None and len(ring_df) > 0:
        for _, r in ring_df.head(10).iterrows():
            if "ring_path_signature" in ring_df.columns:
                path = r["ring_path_signature"].split("->")
                ring_paths.append(path)

    nodes_list = list(nodes_set)[:200]
    edges_list = edges[:500]

    nodes_json = json.dumps([{"id": n, "label": n[:8]} for n in nodes_list])
    edges_json = json.dumps([{"from": e["from"], "to": e["to"]} for e in edges_list])
    rings_json = json.dumps(ring_paths[:10])

    graph_html = f"""
    <div id="network-graph" style="width: 100%; height: 550px; background: {theme["surface"]}; border-radius: 18px; position: relative; border: 1px solid {theme["line"]};">
        <canvas id="graphCanvas" style="width: 100%; height: 100%;"></canvas>
        <div id="graph-legend" style="position: absolute; top: 12px; right: 12px; background: {theme["surface"]}; padding: 10px 14px; border-radius: 12px; border: 1px solid {theme["line"]}; font-size: 11px; color: {theme["muted"]}; box-shadow: 0 10px 18px {theme["shadow"]};">
            <div style="margin-bottom: 4px;"><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#D04A02;margin-right:6px;"></span>Node</div>
            <div style="margin-bottom: 4px;"><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#00A3A1;margin-right:6px;"></span>Ring</div>
            <div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#E0457B;margin-right:6px;"></span>Ring Path</div>
        </div>
        <div id="graph-info" style="position: absolute; bottom: 12px; left: 12px; background: {theme["surface"]}; padding: 8px 12px; border-radius: 12px; border: 1px solid {theme["line"]}; font-size: 11px; color: {theme["muted"]}; box-shadow: 0 10px 18px {theme["shadow"]};">
            Nodes: {len(nodes_list)} | Edges: {len(edges_list)} | Rings: {len(ring_paths)}
        </div>
    </div>
    <script>
    (function() {{
        const canvas = document.getElementById('graphCanvas');
        const ctx = canvas.getContext('2d');
        const container = canvas.parentElement;

        function resize() {{
            canvas.width = container.offsetWidth;
            canvas.height = container.offsetHeight;
        }}
        resize();
        window.addEventListener('resize', resize);

        const nodesData = {nodes_json};
        const edgesData = {edges_json};
        const ringsData = {rings_json};

        const ringNodeSet = new Set();
        ringsData.forEach(ring => ring.forEach(n => ringNodeSet.add(n)));

        const nodeMap = {{}};
        const W = canvas.width;
        const H = canvas.height;
        const cx = W / 2;
        const cy = H / 2;

        nodesData.forEach((n, i) => {{
            const angle = (2 * Math.PI * i) / nodesData.length;
            const radius = Math.min(W, H) * 0.35;
            const jitter = (Math.random() - 0.5) * 60;
            nodeMap[n.id] = {{
                x: cx + Math.cos(angle) * (radius + jitter),
                y: cy + Math.sin(angle) * (radius + jitter),
                label: n.label,
                isRing: ringNodeSet.has(n.id),
                vx: 0,
                vy: 0
            }};
        }});

        function forceSimulation(iterations) {{
            const nodes = Object.values(nodeMap);
            const k = Math.sqrt((W * H) / nodes.length) * 0.5;

            for (let iter = 0; iter < iterations; iter++) {{
                for (let i = 0; i < nodes.length; i++) {{
                    for (let j = i + 1; j < nodes.length; j++) {{
                        let dx = nodes[j].x - nodes[i].x;
                        let dy = nodes[j].y - nodes[i].y;
                        let dist = Math.sqrt(dx * dx + dy * dy) || 1;
                        let force = (k * k) / dist * 0.01;
                        let fx = (dx / dist) * force;
                        let fy = (dy / dist) * force;
                        nodes[i].vx -= fx;
                        nodes[i].vy -= fy;
                        nodes[j].vx += fx;
                        nodes[j].vy += fy;
                    }}
                }}

                edgesData.forEach(e => {{
                    const a = nodeMap[e.from];
                    const b = nodeMap[e.to];
                    if (!a || !b) return;
                    let dx = b.x - a.x;
                    let dy = b.y - a.y;
                    let dist = Math.sqrt(dx * dx + dy * dy) || 1;
                    let force = (dist - k) * 0.003;
                    let fx = (dx / dist) * force;
                    let fy = (dy / dist) * force;
                    a.vx += fx;
                    a.vy += fy;
                    b.vx -= fx;
                    b.vy -= fy;
                }});

                nodes.forEach(n => {{
                    n.x += n.vx * 0.5;
                    n.y += n.vy * 0.5;
                    n.vx *= 0.85;
                    n.vy *= 0.85;
                    n.x = Math.max(30, Math.min(W - 30, n.x));
                    n.y = Math.max(30, Math.min(H - 30, n.y));
                }});
            }}
        }}

        forceSimulation(80);

        function draw() {{
            ctx.clearRect(0, 0, W, H);

            edgesData.forEach(e => {{
                const a = nodeMap[e.from];
                const b = nodeMap[e.to];
                if (!a || !b) return;
                ctx.beginPath();
                ctx.moveTo(a.x, a.y);
                ctx.lineTo(b.x, b.y);
                ctx.strokeStyle = '{theme["graph_edge"]}';
                ctx.lineWidth = 0.8;
                ctx.stroke();
            }});

            ringsData.forEach(ring => {{
                if (ring.length < 2) return;
                ctx.beginPath();
                for (let i = 0; i < ring.length; i++) {{
                    const node = nodeMap[ring[i]];
                    if (!node) continue;
                    if (i === 0) ctx.moveTo(node.x, node.y);
                    else ctx.lineTo(node.x, node.y);
                }}
                const firstNode = nodeMap[ring[0]];
                if (firstNode) ctx.lineTo(firstNode.x, firstNode.y);
                ctx.strokeStyle = '#E0457B';
                ctx.lineWidth = 2;
                ctx.shadowColor = '#E0457B';
                ctx.shadowBlur = 8;
                ctx.stroke();
                ctx.shadowBlur = 0;
            }});

            Object.values(nodeMap).forEach(n => {{
                ctx.beginPath();
                const radius = n.isRing ? 5 : 3;
                ctx.arc(n.x, n.y, radius, 0, Math.PI * 2);

                if (n.isRing) {{
                    ctx.fillStyle = '#E0457B';
                    ctx.shadowColor = '#E0457B';
                    ctx.shadowBlur = 10;
                }} else {{
                    ctx.fillStyle = '#D04A02';
                    ctx.shadowBlur = 0;
                }}
                ctx.fill();
                ctx.shadowBlur = 0;
            }});
        }}

        draw();
    }})();
    </script>
    """

    components.html(graph_html, height=570, scrolling=False)


def node_type_from_id(node_id):
    text = str(node_id)
    prefix = text.split("::", 1)[0]
    return {
        "cust": "Customer",
        "acct": "Account",
        "dev": "Device",
        "ip": "IP",
        "cp": "Counterparty",
        "mch": "Merchant",
    }.get(prefix, prefix.title() if prefix else "Node")


def node_id_display(node_id):
    text = str(node_id)
    if "::" in text:
        _, raw = text.split("::", 1)
        return raw
    return text


def build_interactive_network_figure(df, graph_features=None, ring_df=None, focus_node=None, depth=1, max_nodes=80, color_by="Node Type"):
    theme = current_theme()
    graph = nx.Graph()
    sample = df.head(min(len(df), 800)).copy()
    prefix_map = {
        "customer_id": "cust",
        "account_id": "acct",
        "counterparty_id": "cp",
        "device_id": "dev",
        "merchant_id": "mch",
        "device_ip_address": "ip",
        "ip_address": "ip",
    }

    def node_name(column, value):
        return f"{prefix_map.get(column, column[:4])}::{value}"

    edge_specs = [
        ("customer_id", "account_id"),
        ("account_id", "counterparty_id"),
        ("customer_id", "device_id"),
        ("customer_id", "merchant_id"),
        ("customer_id", "device_ip_address"),
        ("customer_id", "ip_address"),
    ]

    for _, row in sample.iterrows():
        for left, right in edge_specs:
            if left not in sample.columns or right not in sample.columns:
                continue
            u = row.get(left)
            v = row.get(right)
            if pd.isna(u) or pd.isna(v):
                continue
            u_node = node_name(left, u)
            v_node = node_name(right, v)
            if u_node == v_node:
                continue
            amount = float(row.get("amount", 1.0) or 1.0)
            if graph.has_edge(u_node, v_node):
                graph[u_node][v_node]["weight"] += amount
            else:
                graph.add_edge(u_node, v_node, weight=amount)

    if graph.number_of_nodes() == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No graph data available yet.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(color=theme["muted"], size=14),
        )
        fig.update_layout(
            height=520,
            paper_bgcolor=theme["surface"],
            plot_bgcolor=theme["surface"],
            margin=dict(l=20, r=20, t=20, b=20),
        )
        return fig

    if focus_node and focus_node in graph:
        nodes_to_keep = {focus_node}
        frontier = {focus_node}
        for _ in range(max(1, int(depth))):
            next_frontier = set()
            for node in frontier:
                next_frontier.update(graph.neighbors(node))
            next_frontier -= nodes_to_keep
            nodes_to_keep.update(next_frontier)
            frontier = next_frontier
            if not frontier:
                break
        graph = graph.subgraph(nodes_to_keep).copy()

    if graph.number_of_nodes() > int(max_nodes):
        degree_sorted = sorted(graph.degree, key=lambda item: item[1], reverse=True)
        keep_nodes = [node for node, _ in degree_sorted[: int(max_nodes)]]
        graph = graph.subgraph(keep_nodes).copy()

    graph_meta = graph_features.set_index("node_id") if isinstance(graph_features, pd.DataFrame) and "node_id" in graph_features.columns else pd.DataFrame()
    ring_nodes = set()
    if isinstance(ring_df, pd.DataFrame) and not ring_df.empty and "ring_members" in ring_df.columns:
        for members in ring_df["ring_members"]:
            if isinstance(members, str):
                ring_nodes.update([node.strip() for node in members.split(",") if node.strip()])
            elif isinstance(members, (list, tuple, set)):
                ring_nodes.update(map(str, members))

    pos = nx.spring_layout(graph, seed=int(CONFIG.get("random_state", 42)), iterations=60)

    edge_x = []
    edge_y = []
    for left, right in graph.edges():
        x0, y0 = pos[left]
        x1, y1 = pos[right]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    color_palette = [
        PWC_COLORS["primary"],
        PWC_COLORS["accent_teal"],
        PWC_COLORS["accent_gold"],
        PWC_COLORS["accent_rose"],
        PWC_COLORS["info"],
        PWC_COLORS["success"],
    ]
    type_colors = {
        "Customer": PWC_COLORS["primary"],
        "Account": PWC_COLORS["accent_teal"],
        "Device": PWC_COLORS["accent_gold"],
        "IP": PWC_COLORS["info"],
        "Counterparty": PWC_COLORS["accent_rose"],
        "Merchant": PWC_COLORS["success"],
    }

    node_x = []
    node_y = []
    node_text = []
    node_hover = []
    node_size = []
    node_color = []

    for idx, node in enumerate(graph.nodes()):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_label = node_id_display(node)
        node_type = node_type_from_id(node)
        degree = graph.degree(node)
        meta = graph_meta.loc[node] if node in graph_meta.index else {}
        pagerank = float(meta.get("graph_pagerank", 0.0)) if isinstance(meta, pd.Series) else 0.0
        community = int(meta.get("graph_community_id", -1)) if isinstance(meta, pd.Series) and pd.notna(meta.get("graph_community_id", -1)) else -1
        component_size = int(meta.get("graph_component_size", 1)) if isinstance(meta, pd.Series) and pd.notna(meta.get("graph_component_size", 1)) else 1
        core_number = int(meta.get("graph_core_number", 0)) if isinstance(meta, pd.Series) and pd.notna(meta.get("graph_core_number", 0)) else 0
        cycle_flag = int(meta.get("graph_cycle_flag", 0)) if isinstance(meta, pd.Series) and pd.notna(meta.get("graph_cycle_flag", 0)) else 0
        degree_centrality = float(meta.get("graph_degree_centrality", 0.0)) if isinstance(meta, pd.Series) else 0.0
        is_ring = node in ring_nodes

        if color_by == "Community" and community >= 0:
            node_color.append(color_palette[community % len(color_palette)])
        elif is_ring:
            node_color.append(PWC_COLORS["accent_rose"])
        elif color_by == "Node Type":
            node_color.append(type_colors.get(node_type, theme["muted"]))
        else:
            node_color.append(PWC_COLORS["primary"] if degree_centrality >= 0.03 else theme["muted"])

        node_size.append(8 + min(18, max(0, degree * 1.8)))
        node_text.append(node_label)
        node_hover.append(
            f"{node_type}: {node_label}<br>"
            f"Degree: {degree}<br>"
            f"Pagerank: {pagerank:.4f}<br>"
            f"Community: {community if community >= 0 else 'n/a'}<br>"
            f"Component size: {component_size}<br>"
            f"Core number: {core_number}<br>"
            f"Cycle flag: {cycle_flag}<br>"
            f"Ring member: {'Yes' if is_ring else 'No'}"
        )

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.8, color=theme["graph_edge"]),
        hoverinfo="none",
        mode="lines",
    )
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        hovertemplate="%{customdata}<extra></extra>",
        customdata=node_hover,
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=1, color=theme["surface"]),
            opacity=0.95,
        ),
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        height=560,
        showlegend=False,
        hovermode="closest",
        paper_bgcolor=theme["surface"],
        plot_bgcolor=theme["surface"],
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        font=dict(color=theme["text"]),
    )
    return fig


def summarize_numeric_frame(df, numeric_cols=None, top_n=12):
    if df is None or df.empty:
        return []
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    summary = []
    for col in numeric_cols[:top_n]:
        series = pd.to_numeric(df[col], errors="coerce")
        summary.append(
            {
                "feature": col,
                "mean": float(series.mean()) if series.notna().any() else 0.0,
                "median": float(series.median()) if series.notna().any() else 0.0,
                "std": float(series.std()) if series.notna().any() else 0.0,
                "missing_pct": float(series.isna().mean() * 100),
            }
        )
    return summary


def infer_feature_role(column_name, dtype_str=""):
    name = str(column_name).lower()
    dtype_str = str(dtype_str).lower()
    if name in {"label", "mule_category"}:
        return "Target"
    if name.startswith("graph_") or "_graph_" in name:
        return "Graph"
    if any(token in name for token in ["sequence", "transition", "velocity", "dormant", "fanout", "shared", "first_time", "behavioral", "hazard", "hmm", "ring"]):
        return "Behavioral"
    if name.startswith("prob_") or "score" in name:
        return "Risk Score"
    if any(token in name for token in ["event_ts", "timestamp", "date", "time"]):
        return "Datetime"
    if any(token in name for token in ["_id", "email", "phone", "address", "hash", "name", "number"]):
        return "Identifier / PII"
    if "int" in dtype_str or "float" in dtype_str or "bool" in dtype_str:
        return "Numeric"
    return "Categorical"


def describe_feature(column_name):
    name = str(column_name).lower()
    exact_map = {
        "label": "Ground-truth mule class used for training and evaluation.",
        "mule_category": "Synthetic target class generated during data creation.",
        "amount": "Transaction value for the event.",
        "event_ts": "Timestamp when the transaction or login event occurred.",
        "channel": "Source channel for the event such as UPI, ATM, BRANCH, MERCHANT, API, or DIGITAL.",
        "transaction_type": "Transaction action or subtype recorded for the event.",
        "transaction_status": "Raw status or response code for the event.",
        "customer_id": "Customer identifier used to group activity and build customer-level sequences.",
        "account_id": "Account identifier linked to the transaction.",
        "device_id": "Device identifier used to analyze device reuse and shared-device behavior.",
        "counterparty_id": "Counterparty or beneficiary identifier used for fan-out and transfer analysis.",
        "merchant_id": "Merchant identifier associated with payment or cashout activity.",
        "graph_pagerank": "Relative network influence score from graph analytics.",
        "graph_degree_centrality": "Normalized number of direct connections in the graph.",
        "graph_clustering": "Local clustering coefficient showing whether neighbors are interconnected.",
        "graph_community_id": "Community assignment from connected components in the graph.",
        "graph_cycle_flag": "Indicator showing whether the node belongs to a cyclic or ring-like component.",
        "ring_count": "Number of detected ring structures linked to the row.",
        "ring_max_risk_score": "Highest risk score among detected rings linked to the row.",
        "sequence_score": "Aggregated behavioral risk score based on first-time, velocity, fan-out, and dormancy signals.",
        "behavioral_risk_score": "Combined behavioral risk score that blends sequence, transition, network, and amount anomalies.",
        "transition_score": "Unusual channel transition score derived from historical channel-to-channel flow.",
        "hazard_score": "Sequence hazard score from the temporal risk model.",
        "hmm_sequence_anomaly_score": "Anomaly score from the HMM-based sequential model.",
        "final_mule_score": "Final alerting score used to rank and tier alerts.",
        "risk_tier": "Low, medium, or high tier derived from the final score.",
        "priority_band": "Operational alert band used for triage prioritization.",
    }
    if name in exact_map:
        return exact_map[name]
    if name.startswith("prob_"):
        return "Model probability for the corresponding class."
    if "customer" in name and "count" in name:
        return "Customer count signal derived from grouped activity."
    if "device" in name and "count" in name:
        return "Device count signal derived from grouped activity."
    if "ip" in name and "count" in name:
        return "IP sharing signal derived from grouped activity."
    if "counterparty" in name and "count" in name:
        return "Counterparty sharing signal derived from grouped activity."
    if "first_time" in name:
        return "Flag showing whether this is the first observed interaction for the entity combination."
    if "shared" in name:
        return "Shared-entity signal used to detect reuse across multiple customers."
    if "dormant" in name:
        return "Dormancy or reactivation signal that can indicate mule activation behavior."
    if "velocity" in name:
        return "Transaction velocity signal derived from burst activity in a short time window."
    if "fanout" in name:
        return "Fan-out signal showing distribution to multiple counterparties."
    if "session" in name:
        return "Session-level feature used to capture grouped activity over time."
    if "time" in name or "date" in name or "timestamp" in name:
        return "Time-based feature used for sequencing and time-window analysis."
    if "amount" in name:
        return "Amount-based feature used for value and outlier analysis."
    if "score" in name:
        return "Derived risk score used to surface suspicious behavior."
    if "_id" in name:
        return "Identifier column used for joins, grouping, or traceability."
    return "Engineered feature used in the mule risk pipeline."


def build_feature_store_table(df, limit=300):
    if df is None or df.empty:
        return pd.DataFrame(columns=["feature", "dtype", "role", "description", "missing_pct", "unique", "sample_value"])

    rows = []
    for col in df.columns[:limit]:
        series = df[col]
        dtype_str = str(series.dtype)
        non_null = series.dropna()
        sample_value = ""
        if len(non_null) > 0:
            sample_value = str(non_null.iloc[0])
            if len(sample_value) > 80:
                sample_value = sample_value[:77] + "..."
        rows.append(
            {
                "feature": col,
                "dtype": dtype_str,
                "role": infer_feature_role(col, dtype_str),
                "description": describe_feature(col),
                "missing_pct": round(float(series.isna().mean() * 100), 2),
                "unique": int(series.nunique(dropna=True)),
                "sample_value": sample_value,
            }
        )
    return pd.DataFrame(rows)


def render_eda_snapshot(df, title="EDA Snapshot"):
    if df is None or df.empty:
        st.info("Run EDA and imputation to see the snapshot.")
        return

    render_section(title, "analytics")
    render_section_note("Use the tabs to review data quality first, then inspect distributions and the feature inventory before moving into model training.")
    cols = st.columns(4)
    with cols[0]:
        render_kpi("dataset", "Rows", f"{len(df):,}", "", "orange")
    with cols[1]:
        render_kpi("view_column", "Columns", f"{df.shape[1]:,}", "", "teal")
    with cols[2]:
        missing_pct = round(float(df.isna().mean().mean() * 100), 2) if len(df.columns) else 0.0
        render_kpi("report", "Avg Missing", f"{missing_pct:.1f}%", "", "gold")
    with cols[3]:
        numeric_count = len(df.select_dtypes(include=[np.number]).columns)
        render_kpi("functions", "Numeric", f"{numeric_count:,}", "", "green")

    eda_tab1, eda_tab2, eda_tab3 = st.tabs(["Data Quality", "Distributions", "Feature Store"])

    with eda_tab1:
        quality_cols = st.columns([1.0, 1.2])
        with quality_cols[0]:
            missing_cols = df.isna().mean().sort_values(ascending=False).head(20).reset_index()
            missing_cols.columns = ["Column", "Missing %"]
            missing_cols["Missing %"] = (missing_cols["Missing %"] * 100).round(2)
            fig = px.bar(missing_cols, x="Column", y="Missing %", title="Top missing columns")
            fig = apply_dark_theme(fig)
            fig.update_layout(height=360, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with quality_cols[1]:
            summary = summarize_dataframe(df, max_numeric_cols=12)
            if not summary.empty:
                st.dataframe(summary.round(4), use_container_width=True, height=360)
            else:
                st.info("No numeric columns available for a quick summary.")

    with eda_tab2:
        dist_cols = st.columns([1.0, 1.0])
        with dist_cols[0]:
            if "channel" in df.columns:
                channel_df = df["channel"].astype(str).value_counts().reset_index()
                channel_df.columns = ["Channel", "Count"]
                fig = px.bar(channel_df, x="Channel", y="Count", color="Channel", title="Channel distribution")
                fig = apply_dark_theme(fig)
                fig.update_layout(height=340, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            if "mule_category" in df.columns:
                category_df = df["mule_category"].astype(str).value_counts().reset_index().head(15)
                category_df.columns = ["Category", "Count"]
                fig = px.bar(category_df, x="Category", y="Count", color="Category", title="Category distribution")
                fig = apply_dark_theme(fig)
                fig.update_layout(height=340, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        with dist_cols[1]:
            if "amount" in df.columns:
                sample_size = min(len(df), 4000)
                amount_sample = df[["amount"]].sample(sample_size, random_state=CONFIG["random_state"]) if len(df) > sample_size else df[["amount"]]
                fig = px.histogram(amount_sample, x="amount", nbins=40, title="Amount distribution")
                fig = apply_dark_theme(fig)
                fig.update_layout(height=340)
                st.plotly_chart(fig, use_container_width=True)
            if "event_ts" in df.columns:
                ts = pd.to_datetime(df["event_ts"], errors="coerce").dropna()
                if len(ts) > 0:
                    ts_df = ts.dt.floor("D").value_counts().sort_index().reset_index()
                    ts_df.columns = ["Date", "Count"]
                    fig = px.line(ts_df, x="Date", y="Count", markers=True, title="Activity by day")
                    fig = apply_dark_theme(fig)
                    fig.update_layout(height=340)
                    st.plotly_chart(fig, use_container_width=True)

    with eda_tab3:
        with st.expander("Feature Store", expanded=True):
            render_section_note("Each row explains the feature role, data type, missingness, uniqueness, and a sample value for quick review.")
            store_df = build_feature_store_table(df)
            store_cols = st.columns([1.0, 0.9, 1.0])
            with store_cols[0]:
                search = st.text_input("Search feature", value="", key="eda_feature_store_search")
            with store_cols[1]:
                role_filter = st.multiselect(
                    "Role",
                    options=sorted(store_df["role"].unique().tolist()),
                    default=sorted(store_df["role"].unique().tolist()),
                    key="eda_feature_store_role",
                )
            with store_cols[2]:
                max_rows = st.slider("Rows", 25, 300, 120, step=25, key="eda_feature_store_rows")

            view = store_df.copy()
            if search:
                view = view[view["feature"].str.contains(search, case=False, na=False)]
            if role_filter:
                view = view[view["role"].isin(role_filter)]

            role_counts = view["role"].value_counts().reset_index()
            role_counts.columns = ["Role", "Count"]
            if len(role_counts) > 0:
                fig = px.bar(role_counts, x="Role", y="Count", color="Role", title="Feature roles")
                fig = apply_dark_theme(fig)
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            selected_feature = st.selectbox(
                "Inspect feature",
                options=view["feature"].tolist()[:max_rows] if len(view) > 0 else [],
                index=0 if len(view) > 0 else None,
                key="eda_feature_store_inspect",
            ) if len(view) > 0 else None

            if selected_feature:
                selected_row = view[view["feature"] == selected_feature].iloc[0]
                info_cols = st.columns(4)
                with info_cols[0]:
                    render_kpi("view_column", "Role", selected_row["role"], "", "orange")
                with info_cols[1]:
                    render_kpi("description", "Type", selected_row["dtype"], "", "teal")
                with info_cols[2]:
                    render_kpi("report", "Missing", f'{selected_row["missing_pct"]:.2f}%', "", "gold")
                with info_cols[3]:
                    render_kpi("dataset", "Unique", f'{selected_row["unique"]:,}', "", "green")

                with st.expander("Feature description", expanded=True):
                    st.write(selected_row["description"])
                    st.caption(f"Sample value: {selected_row['sample_value']}")

            st.dataframe(view.head(max_rows), use_container_width=True, height=360)


def apply_demo_profile(profile_name):
    profile = CONFIG.get("demo_profiles", {}).get(profile_name)
    if not profile:
        return

    CONFIG["run_profile"] = profile.get("run_profile", profile_name)
    CONFIG["entity_counts"] = profile.get("entity_counts", CONFIG.get("entity_counts", {})).copy()
    for key in [
        "records_per_category_min",
        "records_per_category_max",
        "graph_sample_size",
        "ring_sample_size",
        "max_rings",
        "classifier_estimators",
        "classifier_max_depth",
        "classifier_min_samples_leaf",
    ]:
        if key in profile:
            CONFIG[key] = profile[key]
    st.session_state.demo_profile = profile_name


def apply_manual_config(seed, n_customers, n_accounts, n_devices, records_min, records_max):
    CONFIG["random_state"] = int(seed)
    CONFIG["entity_counts"] = {
        "customers": int(n_customers),
        "accounts": int(n_accounts),
        "devices": int(n_devices),
        "merchants": max(300, int(n_accounts // 5)),
        "counterparties": max(800, int(n_accounts // 2)),
        "video_sessions": max(1200, int(n_devices * 1.1)),
    }
    CONFIG["records_per_category_min"] = int(records_min)
    CONFIG["records_per_category_max"] = max(int(records_min), int(records_max))


def reset_workspace():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


def collect_overview_metrics():
    metrics = {"events": 0, "ring_candidates": 0, "alerts": 0, "macro_f1": None}
    if st.session_state.events is not None:
        metrics["events"] = len(st.session_state.events)
    if st.session_state.ring_df is not None and isinstance(st.session_state.ring_df, pd.DataFrame):
        metrics["ring_candidates"] = len(st.session_state.ring_df)
    if st.session_state.alert_output is not None and isinstance(st.session_state.alert_output, pd.DataFrame):
        metrics["alerts"] = len(st.session_state.alert_output)
    if st.session_state.y_test is not None and st.session_state.test_pred is not None:
        from sklearn.metrics import f1_score

        metrics["macro_f1"] = f1_score(st.session_state.y_test, st.session_state.test_pred, average="macro")
    return metrics


def build_inference_splits(df):
    from multiclass_model import MulticlassModel

    modeler = MulticlassModel()
    train_df, valid_df, test_df = modeler.split_time_based(df, CONFIG["train_end"], CONFIG["valid_end"])

    if min(len(train_df), len(valid_df), len(test_df)) == 0:
        ordered = df.sort_values("event_ts").reset_index(drop=True).copy()
        total = len(ordered)
        if total < 3:
            return ordered.copy(), ordered.head(0).copy(), ordered.copy()
        train_cut = max(1, int(total * 0.60))
        valid_cut = max(train_cut + 1, int(total * 0.80))
        if valid_cut >= total:
            valid_cut = total - 1
        train_df = ordered.iloc[:train_cut].copy()
        valid_df = ordered.iloc[train_cut:valid_cut].copy()
        test_df = ordered.iloc[valid_cut:].copy()
    return train_df, valid_df, test_df


def generate_fresh_inference_batch(seed):
    import random

    from data_ingestion import DataIngestion
    from entity_resolution import EntityResolution
    from feature_engineering import FeatureEngineering
    from graph_analytics import GraphAnalytics
    from sequence_models import SequenceModels

    original_seed = int(CONFIG["random_state"])
    timings = {}

    def timed(label, callback):
        start = time.time()
        result = callback()
        timings[label] = round(time.time() - start, 2)
        return result

    try:
        CONFIG["random_state"] = int(seed)
        np.random.seed(int(seed))
        random.seed(int(seed))

        ingestion = DataIngestion()
        raw_tables = timed("Raw Tables", ingestion.generate_raw_tables)
        txn_tables = timed("Txn Tables", lambda: ingestion.generate_transaction_tables(raw_tables))

        entity = EntityResolution()
        entity_views = timed("Entity Views", lambda: entity.build_entity_views(raw_tables))
        events = timed("Unified Events", lambda: entity.build_unified_events(txn_tables))
        single_view = timed("Single View", lambda: entity.build_single_view(events, entity_views))

        feat = FeatureEngineering()
        clean_df = timed("EDA", lambda: feat.run_eda_and_imputation(single_view))
        feature_df = timed("Features", lambda: feat.feature_engineering(clean_df))

        graph = GraphAnalytics()
        graph_feature_df, graph_features = timed("Graph", lambda: graph.model1_graph_analytics(feature_df))
        graph_feature_df, ring_df = timed("Rings", lambda: graph.model2_ring_detection(graph_feature_df))

        train_df, valid_df, test_df = timed("Split", lambda: build_inference_splits(graph_feature_df))

        seq = SequenceModels()
        model3, train_df, valid_df, test_df = timed("Hazard", lambda: seq.model3_hazard(train_df, valid_df, test_df))
        model4, train_df, valid_df, test_df = timed("HMM", lambda: seq.model4_hmm(train_df, valid_df, test_df))
        model5_outputs = timed("Sequence", lambda: seq.model5_lstm_and_transformer(train_df, valid_df, test_df))

        return {
            "seed": int(seed),
            "profile": st.session_state.demo_profile,
            "generated_at": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            "timings": timings,
            "ring_df": ring_df,
            "train_df": train_df,
            "valid_df": valid_df,
            "test_df": test_df,
            "feature_count": int(feature_df.shape[1]),
            "ring_count": int(len(ring_df)) if ring_df is not None else 0,
            "sequence_model_type": model5_outputs.get("sequence_model_type", "unknown") if isinstance(model5_outputs, dict) else "unknown",
        }
    finally:
        CONFIG["random_state"] = original_seed
        np.random.seed(original_seed)
        random.seed(original_seed)


def get_active_model_bundle():
    if st.session_state.model6_artifacts is not None and st.session_state.feature_cols is not None:
        artifacts = st.session_state.model6_artifacts
        return {
            "source": "Current workspace",
            "model_label": CONFIG.get("classifier_model", "Random Forest"),
            "preprocessor": artifacts.preprocessor,
            "label_encoder": artifacts.label_encoder,
            "base_model": artifacts.base_model,
            "calibrated_model": artifacts.calibrated_model,
            "feature_names": artifacts.feature_names,
            "feature_cols": list(st.session_state.feature_cols),
        }

    if os.path.exists("best_model.pkl"):
        with open("best_model.pkl", "rb") as model_file:
            saved = pickle.load(model_file)
        return {
            "source": "Saved artifact",
            "model_label": saved.get("model_type", "Saved champion model"),
            "preprocessor": saved["preprocessor"],
            "label_encoder": saved["label_encoder"],
            "base_model": saved["base_model"],
            "calibrated_model": saved["calibrated_model"],
            "feature_names": saved["feature_names"],
            "feature_cols": list(saved["feature_cols"]),
        }

    return None


def extract_preprocessor_columns(preprocessor):
    numeric_cols, categorical_cols = [], []
    for name, _transformer, cols in getattr(preprocessor, "transformers_", []):
        if name == "num":
            numeric_cols = list(cols)
        elif name == "cat":
            categorical_cols = list(cols)
    return numeric_cols, categorical_cols


def prepare_inference_features(df, bundle):
    numeric_cols, categorical_cols = extract_preprocessor_columns(bundle["preprocessor"])
    feature_cols = list(bundle["feature_cols"])
    prepared_cols = {}

    for col in feature_cols:
        if col in categorical_cols:
            if col in df.columns:
                prepared_cols[col] = df[col].astype("object").where(df[col].notna(), "MISSING").astype(str)
            else:
                prepared_cols[col] = pd.Series(["MISSING"] * len(df), index=df.index, dtype="object")
        else:
            if col in df.columns:
                prepared_cols[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                prepared_cols[col] = pd.Series(np.nan, index=df.index, dtype="float64")

    for col in numeric_cols:
        if col not in prepared_cols:
            prepared_cols[col] = pd.Series(np.nan, index=df.index, dtype="float64")
    for col in categorical_cols:
        if col not in prepared_cols:
            prepared_cols[col] = pd.Series(["MISSING"] * len(df), index=df.index, dtype="object")

    prepared = pd.DataFrame(prepared_cols, index=df.index)
    return prepared[bundle["feature_cols"]]


def run_multiclass_inference(bundle, scoring_df):
    prepared = prepare_inference_features(scoring_df, bundle)
    processed = bundle["preprocessor"].transform(prepared)
    probabilities = bundle["calibrated_model"].predict_proba(processed)
    pred_idx = np.argmax(probabilities, axis=1)
    pred_labels = bundle["label_encoder"].inverse_transform(pred_idx)
    confidence = probabilities.max(axis=1)
    top_order = np.argsort(probabilities, axis=1)[:, ::-1][:, :3]
    classes = bundle["label_encoder"].classes_

    top_summaries = []
    for row_idx in range(len(scoring_df)):
        parts = []
        for cls_idx in top_order[row_idx]:
            parts.append(f"{classes[cls_idx]} ({probabilities[row_idx, cls_idx]:.2f})")
        top_summaries.append(" | ".join(parts))

    scored = scoring_df.copy()
    scored["predicted_label"] = pred_labels
    scored["prediction_confidence"] = np.round(confidence, 4)
    scored["top_predictions"] = top_summaries

    actual_encoded = None
    report_df = None
    confusion_df = None
    accuracy = None
    macro_f1 = None
    if "label" in scored.columns:
        actual_labels = scored["label"].astype(str).values
        if np.isin(actual_labels, classes).all():
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

            actual_encoded = bundle["label_encoder"].transform(actual_labels)
            accuracy = accuracy_score(actual_encoded, pred_idx)
            macro_f1 = f1_score(actual_encoded, pred_idx, average="macro")
            report_df = pd.DataFrame(
                classification_report(
                    actual_encoded,
                    pred_idx,
                    target_names=classes,
                    output_dict=True,
                    zero_division=0,
                )
            ).transpose()
            confusion_df = pd.DataFrame(
                confusion_matrix(actual_encoded, pred_idx),
                index=classes,
                columns=classes,
            )
            scored["is_correct"] = scored["label"].astype(str) == scored["predicted_label"]

    return {
        "prepared": prepared,
        "processed": processed,
        "probabilities": probabilities,
        "pred_idx": pred_idx,
        "pred_labels": pred_labels,
        "scored_df": scored,
        "report_df": report_df,
        "confusion_df": confusion_df,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "actual_encoded": actual_encoded,
    }


def compute_lime_explanation(processed_matrix, bundle, row_position, predicted_class_idx, num_features=10):
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except Exception as exc:
        raise RuntimeError("LIME is not installed in the active environment.") from exc

    row_position = int(row_position)
    sample_size = min(int(processed_matrix.shape[0]), 300)
    if sample_size <= 0:
        return pd.DataFrame(columns=["feature", "weight", "direction"])

    rng = np.random.default_rng(CONFIG["random_state"])
    if processed_matrix.shape[0] > sample_size:
        sample_idx = rng.choice(processed_matrix.shape[0], size=sample_size, replace=False)
    else:
        sample_idx = np.arange(processed_matrix.shape[0])

    if hasattr(processed_matrix, "toarray"):
        background = processed_matrix[sample_idx].toarray()
        instance = processed_matrix[row_position].toarray()[0]
    else:
        dense = np.asarray(processed_matrix)
        background = dense[sample_idx]
        instance = dense[row_position]

    explainer = LimeTabularExplainer(
        training_data=background,
        feature_names=[str(item) for item in bundle["feature_names"]],
        class_names=[str(item) for item in bundle["label_encoder"].classes_],
        mode="classification",
        discretize_continuous=True,
        random_state=CONFIG["random_state"],
    )
    explanation = explainer.explain_instance(
        instance,
        bundle["calibrated_model"].predict_proba,
        num_features=min(int(num_features), len(bundle["feature_names"])),
        top_labels=1,
    )
    available_labels = explanation.available_labels()
    target_label = int(predicted_class_idx) if int(predicted_class_idx) in available_labels else available_labels[0]
    lime_df = pd.DataFrame(explanation.as_list(label=target_label), columns=["feature", "weight"])
    lime_df["direction"] = np.where(lime_df["weight"] >= 0, "Supports prediction", "Pushes against prediction")
    return lime_df


def run_demo_pipeline():
    import random

    from alert_engine import AlertEngine
    from data_ingestion import DataIngestion
    from entity_resolution import EntityResolution
    from feature_engineering import FeatureEngineering
    from feedback_loop import FeedbackLoop
    from graph_analytics import GraphAnalytics
    from multiclass_model import MulticlassModel
    from sequence_models import SequenceModels

    np.random.seed(CONFIG["random_state"])
    random.seed(CONFIG["random_state"])
    st.session_state.step_times = {}
    progress = st.progress(0)
    status = st.empty()

    def timed_step(label, progress_value, status_text, callback):
        status.text(status_text)
        start = time.time()
        result = callback()
        st.session_state.step_times[label] = time.time() - start
        progress.progress(progress_value)
        return result

    ingestion = DataIngestion()
    raw_tables = timed_step("Raw Tables", 8, "Generating synthetic raw tables...", ingestion.generate_raw_tables)
    txn_tables = timed_step("Txn Tables", 16, "Generating transaction channels...", lambda: ingestion.generate_transaction_tables(raw_tables))

    entity = EntityResolution()
    entity_views = timed_step("Entity Views", 26, "Building entity views...", lambda: entity.build_entity_views(raw_tables))
    events = timed_step("Unified Events", 34, "Building unified event layer...", lambda: entity.build_unified_events(txn_tables))
    single_view = timed_step("Single View", 42, "Building customer single view...", lambda: entity.build_single_view(events, entity_views))

    feat = FeatureEngineering()
    clean_df = timed_step("EDA", 50, "Running data quality checks...", lambda: feat.run_eda_and_imputation(single_view))
    feature_df = timed_step("Features", 58, "Engineering behavioral signals...", lambda: feat.feature_engineering(clean_df))

    graph = GraphAnalytics()
    graph_feature_df, graph_features = timed_step("Graph", 68, "Computing graph analytics...", lambda: graph.model1_graph_analytics(feature_df))
    graph_feature_df, ring_df = timed_step("Rings", 76, "Detecting dense ring candidates...", lambda: graph.model2_ring_detection(graph_feature_df))

    modeler = MulticlassModel()
    train_df, valid_df, test_df = timed_step(
        "Split",
        81,
        "Preparing train, validation, and test windows...",
        lambda: modeler.split_time_based(graph_feature_df, CONFIG["train_end"], CONFIG["valid_end"]),
    )

    seq = SequenceModels()
    model3, train_df, valid_df, test_df = timed_step("Hazard", 85, "Scoring emerging mule risk...", lambda: seq.model3_hazard(train_df, valid_df, test_df))
    model4, train_df, valid_df, test_df = timed_step("HMM", 88, "Running sequence anomaly detection...", lambda: seq.model4_hmm(train_df, valid_df, test_df))
    model5_outputs = timed_step("Sequence", 91, "Preparing sequence intelligence outputs...", lambda: seq.model5_lstm_and_transformer(train_df, valid_df, test_df))

    model_results = timed_step("Classifier", 96, "Training champion and challenger models...", lambda: modeler.model6_multiclass(train_df, valid_df, test_df))
    (
        model6_artifacts,
        feature_cols,
        _y_valid,
        y_test,
        _valid_prob,
        test_prob,
        _valid_pred,
        test_pred,
        _valid_pred_ch,
        test_pred_ch,
        feature_importance,
    ) = model_results

    alert_engine = AlertEngine()
    test_df = timed_step(
        "Decision",
        98,
        "Running decision engine...",
        lambda: alert_engine.model7_decision_engine(test_df, test_prob, model6_artifacts, model5_outputs),
    )
    alert_output, threshold_tbl, channel_thresholds_tbl, class_thresholds_tbl, _enriched_alert_df, threshold_opt = timed_step(
        "Alerts",
        100,
        "Packaging explainable alerts...",
        lambda: alert_engine.model8_alert_pack(test_df, test_prob, model6_artifacts),
    )

    feedback = FeedbackLoop()
    feedback_outputs = feedback.weak_supervision_and_feedback(graph_feature_df, alert_output)

    st.session_state.raw_tables = raw_tables
    st.session_state.txn_tables = txn_tables
    st.session_state.entity_views = entity_views
    st.session_state.events = events
    st.session_state.single_view = single_view
    st.session_state.clean_df = clean_df
    st.session_state.feature_df = feature_df
    st.session_state.graph_feature_df = graph_feature_df
    st.session_state.graph_features = graph_features
    st.session_state.ring_df = ring_df
    st.session_state.train_df = train_df
    st.session_state.valid_df = valid_df
    st.session_state.test_df = test_df
    st.session_state.model3 = model3
    st.session_state.model4 = model4
    st.session_state.model5_outputs = model5_outputs
    st.session_state.model6_artifacts = model6_artifacts
    st.session_state.feature_importance = feature_importance
    st.session_state.y_test = y_test
    st.session_state.test_prob = test_prob
    st.session_state.test_pred = test_pred
    st.session_state.test_pred_ch = test_pred_ch
    st.session_state.alert_output = alert_output
    st.session_state.feedback_outputs = feedback_outputs
    st.session_state.feature_cols = feature_cols
    st.session_state.threshold_table = threshold_tbl
    st.session_state.channel_thresholds = channel_thresholds_tbl
    st.session_state.class_thresholds = class_thresholds_tbl
    st.session_state.threshold_opt = threshold_opt
    st.session_state.current_step = 7
    st.session_state.run_metadata = {
        "profile": st.session_state.demo_profile,
        "seed": CONFIG["random_state"],
        "generated_at": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "event_count": len(events),
    }

    status.empty()
    progress.empty()


def render_sidebar_v2():
    nav_items = [
        {"label": "Overview", "key": "Overview", "step": 0},
        {"label": "Data Generation", "key": "1. Data Generation", "step": 1},
        {"label": "Entity Resolution", "key": "2. Entity Resolution", "step": 2},
        {"label": "Feature Engineering", "key": "3. Feature Engineering", "step": 3},
        {"label": "Graph Analytics", "key": "4. Graph Analytics", "step": 4},
        {"label": "Model Training", "key": "5. Model Training", "step": 5},
        {"label": "Model Inference", "key": "Model Inference", "step": 0},
        {"label": "Alert Engine", "key": "6. Alert Engine", "step": 6},
        {"label": "Feedback Loop", "key": "7. Feedback Loop", "step": 7},
        {"label": "Export", "key": "8. Export", "step": 8},
        {"label": "Monitoring", "key": "Monitoring", "step": 9},
    ]

    step_done = {
        1: st.session_state.raw_tables is not None,
        2: st.session_state.single_view is not None,
        3: st.session_state.feature_df is not None,
        4: st.session_state.graph_feature_df is not None,
        5: st.session_state.model6_artifacts is not None,
        6: st.session_state.alert_output is not None,
        7: st.session_state.feedback_outputs is not None,
        8: os.path.exists("best_model.pkl"),
        9: False,
    }

    if PWC_LOGO_PATH.exists():
        st.image(str(PWC_LOGO_PATH), width=96)
    st.markdown(
        f"""
        <div style="padding:0.15rem 0 0.75rem 0;">
            <div class="sidebar-brand">PwC {APP_NAME}</div>
            <div class="sidebar-desc">{APP_DESCRIPTION}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    completed = sum(step_done[i] for i in range(1, 9))
    pct = completed / 8
    st.caption(f"Run profile: {st.session_state.demo_profile}")
    st.markdown(
        f"""
        <div style="padding:0 0.35rem 0.5rem 0.35rem; margin-bottom:0.8rem;">
            <div style="display:flex; justify-content:space-between; margin-bottom:0.3rem;">
                <span class="sidebar-meta">Stages</span>
                <span class="sidebar-meta">{completed}/8 complete</span>
            </div>
            <div style="height:6px; background:rgba(255,255,255,0.08); border-radius:999px; overflow:hidden;">
                <div style="height:100%; width:{pct*100}%; background:linear-gradient(90deg, {PWC_COLORS["primary"]}, {PWC_COLORS["accent_gold"]}); border-radius:999px;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Workspace Settings", expanded=True):
        selected_theme = st.radio("Theme", ["Light", "Dark"], index=0 if st.session_state.ui_theme == "Light" else 1, horizontal=True)
        selected_profile = st.selectbox("Run profile", list(CONFIG.get("demo_profiles", {}).keys()), index=list(CONFIG.get("demo_profiles", {}).keys()).index(st.session_state.demo_profile))
        if st.button("Apply Settings", use_container_width=True):
            st.session_state.ui_theme = selected_theme
            apply_demo_profile(selected_profile)
            st.rerun()

    for item in nav_items:
        ready = item["step"] == 0 or step_done.get(item["step"], False)
        label = f"{item['label']}"
        if ready and item["step"] not in (0, 9):
            label += "  READY"
        if st.button(
            label,
            key=f"nav_icon_{item['key']}",
            use_container_width=True,
            type="primary" if st.session_state.page == item["key"] else "secondary",
        ):
            st.session_state.page = item["key"]
            st.rerun()

    with st.expander("Stage Status", expanded=False):
        for item in [entry for entry in nav_items if 1 <= entry["step"] <= 8]:
            done = step_done.get(item["step"], False)
            status = "Ready" if done else "Pending"
            st.markdown(f"`{item['label']}`  {status}")

    st.markdown("---")
    if st.button("Reset Workspace", use_container_width=True):
        reset_workspace()


def page_overview():
    metrics = collect_overview_metrics()
    stats = [
        {"value": st.session_state.demo_profile, "label": "Run Profile", "sub": "Execution scale"},
        {"value": f"{metrics['events']:,}" if metrics["events"] else "Not run", "label": "Events", "sub": "Unified activity records"},
        {"value": f"{metrics['alerts']:,}" if metrics["alerts"] else "Pending", "label": "Alerts", "sub": "Scored and packaged"},
        {"value": f"{metrics['macro_f1']:.3f}" if metrics["macro_f1"] is not None else "Pending", "label": "Macro F1", "sub": "Current model quality"},
    ]

    render_top_bar(
        APP_NAME,
        "A streamlined workspace for generating synthetic banking activity, building graph and sequence intelligence, training mule typology models, and packaging explainable alerts.",
        stats,
    )

    profile_names = list(CONFIG.get("demo_profiles", {}).keys())
    current_profile = st.session_state.demo_profile if st.session_state.demo_profile in profile_names else profile_names[0]
    profile_index = profile_names.index(current_profile)

    col_left, col_right = st.columns([1.55, 1.0], gap="large")
    with col_left:
        st.markdown(
            """
            <div class="overview-panel">
                <h3>What this workspace does</h3>
                <p>One run generates realistic multi-channel banking events, resolves customer entities, computes graph and sequence intelligence, trains a multiclass typology model, and packages prioritized alerts with readable reasons.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div style="margin-top:0.9rem;">
                <span class="status-pill ready">Graph analytics</span>
                <span class="status-pill ready">Ring detection</span>
                <span class="status-pill active">Champion vs challenger</span>
                <span class="status-pill active">Alert explanations</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_right:
        selected_profile = st.selectbox("Run profile", profile_names, index=profile_index)
        seed = st.number_input("Seed", value=int(CONFIG["random_state"]), min_value=1, max_value=99999)
        action_col1, action_col2 = st.columns(2)
        with action_col1:
            if st.button("Apply Profile", type="secondary", use_container_width=True):
                apply_demo_profile(selected_profile)
                CONFIG["random_state"] = int(seed)
                st.rerun()
        with action_col2:
            if st.button("Run Pipeline", type="primary", use_container_width=True):
                apply_demo_profile(selected_profile)
                CONFIG["random_state"] = int(seed)
                run_demo_pipeline()
                st.session_state.page = "Overview"
                st.rerun()

    render_section("Workspace Snapshot", "dashboard")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi("timeline", "Events", f"{metrics['events']:,}" if metrics["events"] else "Pending", "Unified customer activity", "orange")
    with col2:
        render_kpi("hub", "Ring Candidates", f"{metrics['ring_candidates']:,}" if metrics["ring_candidates"] else "Pending", "Dense components or cycles", "teal")
    with col3:
        high_risk = 0
        if st.session_state.alert_output is not None and "risk_tier" in st.session_state.alert_output.columns:
            high_risk = int((st.session_state.alert_output["risk_tier"] == "HIGH").sum())
        render_kpi("notification_important", "High Risk", f"{high_risk:,}" if high_risk else "Pending", "Escalation-ready cases", "red")
    with col4:
        exec_minutes = sum(st.session_state.step_times.values()) / 60 if st.session_state.step_times else 0
        render_kpi("timer", "Runtime", f"{exec_minutes:.1f} min" if exec_minutes else "Pending", "Full pipeline execution", "gold")

    details_col1, details_col2 = st.columns([1.4, 1.0], gap="large")
    with details_col1:
        render_section("Latest Run Story", "auto_awesome")
        if st.session_state.alert_output is None:
            st.info("Run the pipeline to populate the workspace, metrics, and drill-down pages.")
        else:
            alert_preview = st.session_state.alert_output.sort_values("final_mule_score", ascending=False).head(15)
            st.dataframe(alert_preview, use_container_width=True, height=360)
    with details_col2:
        render_section("Operational View", "monitoring")
        if st.session_state.step_times:
            timeline = pd.DataFrame([{"Step": key, "Seconds": round(value, 2)} for key, value in st.session_state.step_times.items()])
            fig = px.bar(
                timeline,
                x="Seconds",
                y="Step",
                orientation="h",
                color="Seconds",
                color_continuous_scale=[[0, "#F7E6DA"], [0.5, "#D04A02"], [1, "#FFB600"]],
            )
            fig = apply_dark_theme(fig)
            fig.update_layout(height=max(300, len(timeline) * 28), showlegend=False, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Execution timing will appear here after the first run.")


def init_session_state():
    defaults = {
        "page": "Overview",
        "demo_profile": CONFIG.get("run_profile", "Standard"),
        "ui_theme": "Light",
        "current_step": 0,
        "raw_tables": None,
        "txn_tables": None,
        "entity_views": None,
        "events": None,
        "single_view": None,
        "clean_df": None,
        "feature_df": None,
        "graph_feature_df": None,
        "graph_features": None,
        "ring_df": None,
        "train_df": None,
        "valid_df": None,
        "test_df": None,
        "model3": None,
        "model4": None,
        "model5_outputs": None,
        "model6_artifacts": None,
        "feature_importance": None,
        "y_test": None,
        "test_prob": None,
        "test_pred": None,
        "test_pred_ch": None,
        "alert_output": None,
        "feedback_outputs": None,
        "threshold_table": None,
        "channel_thresholds": None,
        "class_thresholds": None,
        "threshold_opt": None,
        "run_metadata": None,
        "step_times": {},
        "feature_cols": None,
        "inference_batch": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def render_sidebar():
    nav_items = [
        {"label": "Data Generation", "icon": "database", "key": "1. Data Generation", "step": 1},
        {"label": "Entity Resolution", "icon": "hub", "key": "2. Entity Resolution", "step": 2},
        {"label": "Feature Engineering", "icon": "engineering", "key": "3. Feature Engineering", "step": 3},
        {"label": "Graph Analytics", "icon": "share", "key": "4. Graph Analytics", "step": 4},
        {"label": "Model Training", "icon": "model_training", "key": "5. Model Training", "step": 5},
        {"label": "Alert Engine", "icon": "notifications", "key": "6. Alert Engine", "step": 6},
        {"label": "Feedback Loop", "icon": "loop", "key": "7. Feedback Loop", "step": 7},
        {"label": "Export", "icon": "cloud_download", "key": "8. Export", "step": 8},
        {"label": "Monitoring", "icon": "monitoring", "key": "Monitoring", "step": 0},
    ]

    step_done = {
        1: st.session_state.raw_tables is not None,
        2: st.session_state.single_view is not None,
        3: st.session_state.feature_df is not None,
        4: st.session_state.graph_feature_df is not None,
        5: st.session_state.model6_artifacts is not None,
        6: st.session_state.alert_output is not None,
        7: st.session_state.feedback_outputs is not None,
        8: os.path.exists("best_model.pkl"),
    }

    st.markdown(f"""
    <div style="padding: 1.25rem 1rem 0.5rem 1rem; border-bottom: 1px solid #2A2A2A; margin-bottom: 1rem;">
        <div style="display: flex; align-items: center; gap: 0.5rem;">
            <div style="width: 32px; height: 32px; background: {PWC_COLORS["primary"]}; border-radius: 8px;
                display: flex; align-items: center; justify-content: center;">
                <span style="color: white; font-weight: 800; font-size: 0.9rem;">P</span>
            </div>
            <div>
                <div style="font-size: 0.9rem; font-weight: 700; color: #FFFFFF;">PwC</div>
                <div style="font-size: 0.65rem; color: #888888;">Mule Detection</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    completed = sum(step_done.values())
    pct = completed / 8

    st.markdown(f"""
    <div style="padding: 0 1rem; margin-bottom: 1rem;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.3rem;">
            <span style="font-size: 0.7rem; color: #888;">Progress</span>
            <span style="font-size: 0.7rem; color: {PWC_COLORS["primary"]};">{completed}/8</span>
        </div>
        <div style="height: 4px; background: #2A2A2A; border-radius: 2px; overflow: hidden;">
            <div style="height: 100%; width: {pct*100}%; background: {PWC_COLORS["primary"]}; border-radius: 2px;
                transition: width 0.3s ease;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="padding: 0 0.5rem;">', unsafe_allow_html=True)

    for item in nav_items:
        is_active = st.session_state.page == item["key"]
        is_done = step_done.get(item["step"], False)

        if st.button(
            f"{'* ' if is_active else ''}{item['label']}",
            key=f"nav_{item['key']}",
            use_container_width=True,
            type="primary" if is_active else "secondary"
        ):
            st.session_state.page = item["key"]
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    if st.button("Reset All", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


def page_data_generation():
    stats = []
    if st.session_state.raw_tables:
        total_rows = sum(df.shape[0] for df in st.session_state.raw_tables.values())
        stats = [
            {"value": f"{len(st.session_state.raw_tables)}", "label": "Tables"},
            {"value": f"{total_rows:,}", "label": "Records"},
        ]

    render_top_bar("Data Generation", "Generate synthetic banking data for mule detection analysis", stats)
    render_progress_steps(1)
    render_page_intro("Set the synthetic population size, seed, and record ranges, then inspect the raw and transaction tables that feed every downstream stage.")

    render_section("Configuration", "settings")
    render_section_note("Tune the synthetic population and record scale before generating the source tables.")

    entity_counts = CONFIG.get("entity_counts", {})
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        n_customers = st.number_input("Customers", value=int(entity_counts.get("customers", 6500)), min_value=1000, max_value=100000, step=500)
    with col2:
        n_accounts = st.number_input("Accounts", value=int(entity_counts.get("accounts", 8000)), min_value=1000, max_value=100000, step=500)
    with col3:
        n_devices = st.number_input("Devices", value=int(entity_counts.get("devices", 5000)), min_value=1000, max_value=100000, step=500)
    with col4:
        records_min = st.number_input("Min / category", value=int(CONFIG.get("records_per_category_min", 800)), min_value=100, max_value=10000, step=100)
    with col5:
        records_max = st.number_input("Max / category", value=int(CONFIG.get("records_per_category_max", 1100)), min_value=100, max_value=12000, step=100)
    with col6:
        seed = st.number_input("Random Seed", value=int(CONFIG["random_state"]), min_value=1, max_value=99999)

    if st.button("Generate Data", type="primary", use_container_width=False):
        from data_ingestion import DataIngestion

        apply_manual_config(seed, n_customers, n_accounts, n_devices, records_min, records_max)
        np.random.seed(seed)

        progress = st.progress(0)
        status = st.empty()

        status.text("Generating raw tables...")
        ingestion = DataIngestion()
        start = time.time()
        raw_tables = ingestion.generate_raw_tables()
        st.session_state.step_times["Raw Tables"] = time.time() - start
        progress.progress(50)

        status.text("Generating transaction tables...")
        start = time.time()
        txn_tables = ingestion.generate_transaction_tables(raw_tables)
        st.session_state.step_times["Txn Tables"] = time.time() - start
        progress.progress(100)

        st.session_state.raw_tables = raw_tables
        st.session_state.txn_tables = txn_tables
        st.session_state.current_step = max(st.session_state.current_step, 1)

        status.empty()
        progress.empty()
        st.rerun()

    if st.session_state.raw_tables is not None:
        raw_tables = st.session_state.raw_tables
        txn_tables = st.session_state.txn_tables

        total_raw = sum(df.shape[0] for df in raw_tables.values())
        total_txn = sum(df.shape[0] for df in txn_tables.values())

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            render_kpi("storage", "Raw Tables", len(raw_tables), f"{total_raw:,} records", "orange")
        with col2:
            render_kpi("receipt_long", "Txn Tables", len(txn_tables), f"{total_txn:,} records", "teal")
        with col3:
            render_kpi("dataset", "Total Records", f"{total_raw + total_txn:,}", "", "gold")
        with col4:
            t = st.session_state.step_times.get("Raw Tables", 0) + st.session_state.step_times.get("Txn Tables", 0)
            render_kpi("timer", "Gen Time", f"{t:.1f}s", "", "green")
        with col5:
            total_cols = sum(df.shape[1] for df in raw_tables.values()) + sum(df.shape[1] for df in txn_tables.values())
            render_kpi("view_column", "Total Columns", f"{total_cols}", "", "rose")

        render_section("Table Explorer", "table_chart")
        render_section_note("Use the tabs to inspect preview rows, schema, and aggregate statistics for each raw or transaction table.")

        all_tables = {**raw_tables, **txn_tables}
        explorer_cols = st.columns([1.0, 1.0, 1.0])
        with explorer_cols[0]:
            table_query = st.text_input("Search table", value="")
        with explorer_cols[1]:
            table_kind = st.selectbox("Table kind", ["All", "Raw", "Txn"])
        with explorer_cols[2]:
            preview_rows = st.slider("Preview rows", 25, 300, 100, step=25)

        table_names = list(all_tables.keys())
        if table_kind == "Raw":
            table_names = list(raw_tables.keys())
        elif table_kind == "Txn":
            table_names = list(txn_tables.keys())
        if table_query:
            table_names = [name for name in table_names if table_query.lower() in name.lower()]
        if not table_names:
            st.info("No tables match the current filter.")
            return

        selected = st.selectbox("Select table", table_names, label_visibility="collapsed")

        if selected:
            df = all_tables[selected]

            tab1, tab2, tab3 = st.tabs(["Preview", "Schema", "Statistics"])

            with tab1:
                st.dataframe(df.head(preview_rows), use_container_width=True, height=400)
                if "amount" in df.columns:
                    amount_col1, amount_col2 = st.columns(2)
                    with amount_col1:
                        fig = px.histogram(df.sample(min(len(df), 2000), random_state=CONFIG["random_state"]), x="amount", nbins=40, title=f"{selected} amount distribution")
                        fig = apply_dark_theme(fig)
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    with amount_col2:
                        if "channel" in df.columns:
                            fig = px.box(df, x="channel", y="amount", color="channel", title=f"{selected} amount by channel")
                            fig = apply_dark_theme(fig)
                            fig.update_layout(height=300, showlegend=False)
                            st.plotly_chart(fig, use_container_width=True)

            with tab2:
                schema = pd.DataFrame({
                    "Column": df.columns,
                    "Type": df.dtypes.astype(str).values,
                    "Non Null": df.count().values,
                    "Null %": (df.isna().mean() * 100).round(2).values,
                    "Unique": df.nunique().values,
                })
                st.dataframe(schema, use_container_width=True, height=400)

            with tab3:
                summary = df.describe(include="all").transpose().fillna("")
                st.dataframe(summary.round(4), use_container_width=True, height=400)
                numeric_summary = summarize_numeric_frame(df)
                if numeric_summary:
                    st.dataframe(pd.DataFrame(numeric_summary).round(4), use_container_width=True, height=220)


def page_entity_resolution():
    render_top_bar("Entity Resolution", "Build entity views, unified events, and customer single view")
    render_progress_steps(2)
    render_page_intro("We resolve customers, accounts, devices, merchants, and counterparties into reusable entity views, then stitch them into a unified event stream and a single customer-level view.")

    if st.session_state.raw_tables is None:
        st.warning("Complete Step 1 first: Generate Data")
        return

    render_section("Build Pipeline", "play_arrow")
    render_section_note("Run the three actions in order so the entity views, unified events, and single view stay aligned.")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Build Entity Views", type="primary", use_container_width=True):
            from entity_resolution import EntityResolution

            with st.spinner("Building entity views..."):
                start = time.time()
                entity = EntityResolution()
                entity_views = entity.build_entity_views(st.session_state.raw_tables)
                st.session_state.step_times["Entity Views"] = time.time() - start
                st.session_state.entity_views = entity_views
            st.rerun()

    with col2:
        if st.button("Build Unified Events", type="primary", use_container_width=True):
            if st.session_state.entity_views is None:
                st.warning("Build entity views first")
            else:
                from entity_resolution import EntityResolution

                with st.spinner("Building unified events..."):
                    start = time.time()
                    entity = EntityResolution()
                    events = entity.build_unified_events(st.session_state.txn_tables)
                    st.session_state.step_times["Unified Events"] = time.time() - start
                    st.session_state.events = events
                st.rerun()

    with col3:
        if st.button("Build Single View", type="primary", use_container_width=True):
            if st.session_state.events is None:
                st.warning("Build unified events first")
            else:
                from entity_resolution import EntityResolution

                with st.spinner("Building single view..."):
                    start = time.time()
                    entity = EntityResolution()
                    single_view = entity.build_single_view(st.session_state.events, st.session_state.entity_views)
                    st.session_state.step_times["Single View"] = time.time() - start
                    st.session_state.single_view = single_view
                    st.session_state.current_step = max(st.session_state.current_step, 2)
                st.rerun()

    if st.session_state.entity_views is not None:
        render_section("Entity Views", "hub")
        render_section_note("These cards summarize the normalized entity tables created from the raw source data.")

        cols = st.columns(len(st.session_state.entity_views))
        for i, (name, df) in enumerate(st.session_state.entity_views.items()):
            with cols[i]:
                short_name = name.replace("_view", "").title()
                render_kpi("group", short_name, f"{df.shape[0]:,}", f"{df.shape[1]} columns", "orange")

    if st.session_state.events is not None:
        render_section("Unified Events", "timeline")
        render_section_note("The unified stream combines the resolved entities into a single activity feed for downstream analysis.")

        ev = st.session_state.events
        col1, col2, col3 = st.columns(3)
        with col1:
            render_kpi("event", "Total Events", f"{len(ev):,}", "", "teal")
        with col2:
            if "channel" in ev.columns:
                render_kpi("mediation", "Channels", f"{ev['channel'].nunique()}", "", "gold")
        with col3:
            if "customer_id" in ev.columns:
                render_kpi("person", "Customers", f"{ev['customer_id'].nunique():,}", "", "green")

        if "channel" in ev.columns:
            dist = ev["channel"].value_counts().reset_index()
            dist.columns = ["Channel", "Count"]
            fig = px.bar(dist, x="Channel", y="Count", color="Channel",
                         color_discrete_sequence=[PWC_COLORS["primary"], PWC_COLORS["accent_teal"],
                                                  PWC_COLORS["accent_gold"], PWC_COLORS["accent_rose"],
                                                  PWC_COLORS["info"], PWC_COLORS["success"]])
            fig = apply_dark_theme(fig)
            fig.update_layout(showlegend=False, height=350, title="Events by Channel")
            st.plotly_chart(fig, use_container_width=True)

        event_controls = st.columns([1.0, 1.0, 1.0])
        with event_controls[0]:
            top_customers = []
            if "customer_id" in ev.columns:
                top_customers = ev["customer_id"].astype(str).value_counts().head(100).index.tolist()
            selected_customer = st.selectbox("Inspect customer", ["All customers"] + top_customers if top_customers else ["All customers"])
        with event_controls[1]:
            channel_options = ["All channels"]
            if "channel" in ev.columns:
                channel_options += sorted(ev["channel"].astype(str).unique().tolist())
            selected_channel = st.selectbox("Channel", channel_options)
        with event_controls[2]:
            event_window = st.slider("Recent rows", 25, 500, 100, step=25)

        event_view = ev.copy()
        if selected_customer != "All customers" and "customer_id" in event_view.columns:
            event_view = event_view[event_view["customer_id"].astype(str) == str(selected_customer)]
        if selected_channel != "All channels" and "channel" in event_view.columns:
            event_view = event_view[event_view["channel"].astype(str) == str(selected_channel)]

        if not event_view.empty:
            drill_col1, drill_col2 = st.columns(2)
            with drill_col1:
                if "event_ts" in event_view.columns:
                    ts_view = event_view.copy()
                    ts_view["event_day"] = pd.to_datetime(ts_view["event_ts"]).dt.floor("D")
                    if "channel" in ts_view.columns:
                        daily = ts_view.groupby(["event_day", "channel"]).size().reset_index(name="Count")
                        fig = px.line(daily, x="event_day", y="Count", color="channel", markers=True, title="Daily event activity")
                    else:
                        daily = ts_view.groupby("event_day").size().reset_index(name="Count")
                        fig = px.line(daily, x="event_day", y="Count", markers=True, title="Daily event activity")
                    fig = apply_dark_theme(fig)
                    fig.update_layout(height=360)
                    st.plotly_chart(fig, use_container_width=True)
            with drill_col2:
                if "amount" in event_view.columns:
                    fig = px.histogram(event_view, x="amount", nbins=40, title="Event amount distribution")
                    fig = apply_dark_theme(fig)
                    fig.update_layout(height=360)
                    st.plotly_chart(fig, use_container_width=True)

            if "counterparty_id" in event_view.columns:
                counterparty_counts = event_view["counterparty_id"].astype(str).value_counts().head(12).reset_index()
                counterparty_counts.columns = ["Counterparty", "Count"]
                fig = px.bar(counterparty_counts, x="Counterparty", y="Count", title="Top counterparties")
                fig = apply_dark_theme(fig)
                fig.update_layout(height=330, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            st.dataframe(event_view.head(event_window), use_container_width=True, height=280)

    if st.session_state.single_view is not None:
        render_section("Single View Preview", "table_chart")
        render_section_note("This flattened customer view is the feature-ready dataset that feeds the next stage.")
        single_view = st.session_state.single_view
        sv_col1, sv_col2 = st.columns([1.0, 1.0])
        with sv_col1:
            sv_query = st.text_input("Search single view columns", value="")
        with sv_col2:
            sv_rows = st.slider("Single view rows", 25, 250, 100, step=25)
        filtered_sv = single_view
        if sv_query:
            keep_cols = [c for c in single_view.columns if sv_query.lower() in c.lower()]
            if keep_cols:
                filtered_sv = single_view[keep_cols]
        st.dataframe(filtered_sv.head(sv_rows), use_container_width=True, height=300)


def page_feature_engineering():
    render_top_bar("Feature Engineering", "EDA, imputation, and advanced feature generation")
    render_progress_steps(3)
    render_page_intro("Clean the single view, inspect missingness and distributions, then turn the resolved activity into model-ready features.")

    if st.session_state.single_view is None:
        st.warning("Complete Step 2 first: Entity Resolution")
        return

    render_section("Run Controls", "tune")
    render_section_note("Run EDA first to understand data quality, then generate engineered features from the cleaned single view.")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Run EDA and Imputation", type="primary", use_container_width=True):
            from feature_engineering import FeatureEngineering

            with st.spinner("Running EDA and imputation..."):
                start = time.time()
                feat = FeatureEngineering()
                clean_df = feat.run_eda_and_imputation(st.session_state.single_view)
                st.session_state.step_times["EDA"] = time.time() - start
                st.session_state.clean_df = clean_df
            st.rerun()

    with col2:
        if st.button("Generate Features", type="primary", use_container_width=True):
            if st.session_state.clean_df is None:
                st.warning("Run EDA first")
            else:
                from feature_engineering import FeatureEngineering

                with st.spinner("Generating features..."):
                    start = time.time()
                    feat = FeatureEngineering()
                    feature_df = feat.feature_engineering(st.session_state.clean_df)
                    st.session_state.step_times["Features"] = time.time() - start
                st.session_state.feature_df = feature_df
                st.session_state.current_step = max(st.session_state.current_step, 3)
            st.rerun()

    if st.session_state.clean_df is not None:
        render_eda_snapshot(st.session_state.clean_df, "EDA Results")

    if st.session_state.feature_df is not None:
        df = st.session_state.feature_df
        render_section_note("Preview the engineered dataframe, inspect distributions, and browse the feature catalog before training the model.")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            render_kpi("functions", "Total Features", f"{df.shape[1]}", "", "orange")
        with col2:
            render_kpi("pin", "Numeric", f"{len(df.select_dtypes(include=[np.number]).columns)}", "", "teal")
        with col3:
            render_kpi("text_fields", "Categorical", f"{len(df.select_dtypes(include=['object']).columns)}", "", "gold")
        with col4:
            render_kpi("data_array", "Records", f"{df.shape[0]:,}", "", "green")

        tab1, tab2, tab3, tab4 = st.tabs(["Data Preview", "Feature Distributions", "Label Analysis", "Feature Store"])

        with tab1:
            st.caption("Inspect the engineered dataframe directly before drilling into distributions and store metadata.")
            st.dataframe(df.head(200), use_container_width=True, height=400)

        with tab2:
            st.caption("Look at one numeric feature at a time to check shape, spread, and outliers.")
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            sel_options = num_cols[:40]
            sel = st.selectbox("Select feature", sel_options) if sel_options else None
            if sel:
                hist_sample = df[[sel]].sample(min(len(df), 4000), random_state=CONFIG["random_state"]) if len(df) > 4000 else df[[sel]]
                fig = px.histogram(hist_sample, x=sel, nbins=50, color_discrete_sequence=[PWC_COLORS["primary"]])
                fig = apply_dark_theme(fig)
                fig.update_layout(height=400, title=f"Distribution: {sel}")
                st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.caption("Compare label mix and amount behavior to see whether the classes separate cleanly.")
            if "label" in df.columns:
                col1, col2 = st.columns(2)
                with col1:
                    dist = df["label"].value_counts().reset_index()
                    dist.columns = ["Label", "Count"]
                    fig = px.bar(dist, x="Label", y="Count", color="Label")
                    fig = apply_dark_theme(fig)
                    fig.update_layout(height=400, showlegend=False, title="Label Distribution")
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    if "amount" in df.columns:
                        amount_sample = df[["label", "amount"]].sample(min(len(df), 4000), random_state=CONFIG["random_state"]) if len(df) > 4000 else df[["label", "amount"]]
                        fig = px.box(amount_sample, x="label", y="amount", color="label")
                        fig = apply_dark_theme(fig)
                        fig.update_layout(height=400, showlegend=False, title="Amount by Label")
                        st.plotly_chart(fig, use_container_width=True)

        render_section("Interactive Feature Lens", "travel_explore")
        render_section_note("Filter the engineered dataset by label, channel, and amount range, then inspect correlation and trend patterns.")
        lens_top = st.columns([1.1, 1.0, 1.0])

        label_filter = []
        channel_filter = []
        amount_window = None
        if "label" in df.columns:
            with lens_top[0]:
                label_filter = st.multiselect(
                    "Filter labels",
                    options=sorted(df["label"].astype(str).unique().tolist()),
                    default=sorted(df["label"].astype(str).unique().tolist())[:4],
                )
        elif "channel" in df.columns:
            with lens_top[0]:
                channel_filter = st.multiselect(
                    "Filter channels",
                    options=sorted(df["channel"].astype(str).unique().tolist()),
                    default=sorted(df["channel"].astype(str).unique().tolist())[:4],
                )

        if "amount" in df.columns:
            with lens_top[1]:
                amount_window = st.slider(
                    "Amount range",
                    float(df["amount"].min()),
                    float(df["amount"].max()),
                    (float(df["amount"].quantile(0.05)), float(df["amount"].quantile(0.95))),
                )

        trend_metric_options = [c for c in ["amount", "cust_amount_zscore", "hours_since_prev_txn", "sequence_score", "ring_max_risk_score"] if c in df.columns]
        trend_metric = None
        if trend_metric_options:
            with lens_top[2]:
                trend_metric = st.selectbox("Trend metric", trend_metric_options, index=0)

        filtered = df.copy()
        if label_filter:
            filtered = filtered[filtered["label"].astype(str).isin(label_filter)]
        if channel_filter and "channel" in filtered.columns:
            filtered = filtered[filtered["channel"].astype(str).isin(channel_filter)]
        if amount_window and "amount" in filtered.columns:
            filtered = filtered[(filtered["amount"] >= amount_window[0]) & (filtered["amount"] <= amount_window[1])]

        if "event_ts" in filtered.columns and trend_metric:
            trend_df = filtered.copy()
            trend_df["event_day"] = pd.to_datetime(trend_df["event_ts"]).dt.floor("D")
            if "label" in trend_df.columns and trend_df["label"].nunique() <= 8:
                trend = trend_df.groupby(["event_day", "label"])[trend_metric].mean().reset_index()
                fig = px.line(trend, x="event_day", y=trend_metric, color="label", markers=True)
            elif "channel" in trend_df.columns and trend_df["channel"].nunique() <= 8:
                trend = trend_df.groupby(["event_day", "channel"])[trend_metric].mean().reset_index()
                fig = px.line(trend, x="event_day", y=trend_metric, color="channel", markers=True)
            else:
                trend = trend_df.groupby("event_day")[trend_metric].mean().reset_index()
                fig = px.line(trend, x="event_day", y=trend_metric, markers=True)
            fig = apply_dark_theme(fig)
            fig.update_layout(height=380, title=f"{trend_metric} trend")
            st.plotly_chart(fig, use_container_width=True)

        lens_col1, lens_col2 = st.columns(2)
        with lens_col1:
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            corr_defaults = [c for c in ["amount", "cust_amount_zscore", "sequence_score", "ring_max_risk_score", "hours_since_prev_txn"] if c in num_cols]
            corr_defaults = [c for c in corr_defaults if c in num_cols]
            if not corr_defaults:
                corr_defaults = num_cols[: min(8, len(num_cols))]
            corr_choices = st.multiselect(
                "Correlation features",
                options=num_cols[:40],
                default=corr_defaults,
            )
            if len(corr_choices) >= 2:
                corr = filtered[corr_choices].corr().round(2)
                fig = px.imshow(
                    corr,
                    text_auto=True,
                    color_continuous_scale=[[0, "#F7E6DA"], [0.5, "#D04A02"], [1, "#FFB600"]],
                    title="Feature correlation",
                )
                fig = apply_dark_theme(fig)
                fig.update_layout(height=520)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Choose at least two numeric features for correlation view.")

        with lens_col2:
            scatter_x = "amount" if "amount" in filtered.columns else None
            scatter_y = "cust_amount_zscore" if "cust_amount_zscore" in filtered.columns else None
            if scatter_x and scatter_y:
                point_color = "label" if "label" in filtered.columns else ("channel" if "channel" in filtered.columns else None)
                sample_size = min(len(filtered), 2000)
                plot_df = filtered.sample(sample_size, random_state=CONFIG["random_state"]) if len(filtered) > sample_size else filtered
                fig = px.scatter(
                    plot_df,
                    x=scatter_x,
                    y=scatter_y,
                    color=point_color,
                    hover_data=[c for c in ["customer_id", "account_id", "channel", "label", "sequence_score"] if c in plot_df.columns],
                    title="Amount vs behavioural z-score",
                    opacity=0.7,
                )
                fig = apply_dark_theme(fig)
                fig.update_layout(height=520)
                st.plotly_chart(fig, use_container_width=True)

        st.caption(f"Filtered rows: {len(filtered):,} of {len(df):,}")
        st.dataframe(filtered.head(200), use_container_width=True, height=280)

    with tab4:
            with st.expander("Feature Store", expanded=True):
                st.caption("Browse the feature catalog, its role in the pipeline, and a short explanation of what each column represents.")
                feature_store_df = build_feature_store_table(df)
                store_filter_cols = st.columns([1.0, 0.9, 0.9])
                with store_filter_cols[0]:
                    store_search = st.text_input("Search feature", value="", key="feature_store_search")
                with store_filter_cols[1]:
                    store_roles = st.multiselect(
                        "Role",
                        options=sorted(feature_store_df["role"].unique().tolist()),
                        default=sorted(feature_store_df["role"].unique().tolist()),
                        key="feature_store_roles",
                    )
                with store_filter_cols[2]:
                    store_rows = st.slider("Rows", 25, 300, 120, step=25, key="feature_store_rows")

                store_view = feature_store_df.copy()
                if store_search:
                    store_view = store_view[store_view["feature"].str.contains(store_search, case=False, na=False)]
                if store_roles:
                    store_view = store_view[store_view["role"].isin(store_roles)]

                role_counts = store_view["role"].value_counts().reset_index()
                role_counts.columns = ["Role", "Count"]
                if not role_counts.empty:
                    fig = px.bar(role_counts, x="Role", y="Count", color="Role", title="Feature roles")
                    fig = apply_dark_theme(fig)
                    fig.update_layout(height=300, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

                selected_feature = st.selectbox(
                    "Inspect feature",
                    options=store_view["feature"].tolist()[:store_rows] if len(store_view) > 0 else [],
                    index=0 if len(store_view) > 0 else None,
                    key="feature_store_inspect",
                ) if len(store_view) > 0 else None

                if selected_feature:
                    selected_row = store_view[store_view["feature"] == selected_feature].iloc[0]
                    info_cols = st.columns(4)
                    with info_cols[0]:
                        render_kpi("view_column", "Role", selected_row["role"], "", "orange")
                    with info_cols[1]:
                        render_kpi("description", "Type", selected_row["dtype"], "", "teal")
                    with info_cols[2]:
                        render_kpi("report", "Missing", f'{selected_row["missing_pct"]:.2f}%', "", "gold")
                    with info_cols[3]:
                        render_kpi("dataset", "Unique", f'{selected_row["unique"]:,}', "", "green")

                    with st.expander("Feature description", expanded=True):
                        st.write(selected_row["description"])
                        st.caption(f"Sample value: {selected_row['sample_value']}")

                st.dataframe(store_view.head(store_rows), use_container_width=True, height=360)


def page_graph_analytics():
    render_top_bar("Graph Analytics", "Network analysis, community detection, and mule ring identification")
    render_progress_steps(4)
    render_page_intro("Build the transaction graph, surface dense communities and cycles, and inspect suspicious rings through an interactive network explorer.")

    if st.session_state.feature_df is None:
        st.warning("Complete Step 3 first: Feature Engineering")
        return

    render_section("Configuration", "tune")
    render_section_note("Set the graph sample size and ring search bounds before running community detection and cycle analysis.")

    col1, col2, col3 = st.columns(3)
    with col1:
        graph_sample = st.number_input("Graph Sample Size", value=int(CONFIG.get("graph_sample_size", 8000)), min_value=1000, max_value=100000, step=1000)
    with col2:
        ring_sample = st.number_input("Ring Sample Size", value=int(CONFIG.get("ring_sample_size", 3000)), min_value=500, max_value=100000, step=500)
    with col3:
        max_rings = st.number_input("Max Rings", value=int(CONFIG.get("max_rings", 25)), min_value=5, max_value=1000, step=5)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Run Graph Analytics", type="primary", use_container_width=True):
            from graph_analytics import GraphAnalytics

            CONFIG["graph_sample_size"] = int(graph_sample)
            with st.spinner("Running graph analytics..."):
                start = time.time()
                graph = GraphAnalytics()
                graph_df, graph_features = graph.model1_graph_analytics(st.session_state.feature_df)
                st.session_state.step_times["Graph"] = time.time() - start
                st.session_state.graph_feature_df = graph_df
                st.session_state.graph_features = graph_features
            st.rerun()

    with col2:
        if st.button("Run Ring Detection", type="primary", use_container_width=True):
            if st.session_state.graph_feature_df is None:
                st.warning("Run graph analytics first")
            else:
                from graph_analytics import GraphAnalytics

                CONFIG["ring_sample_size"] = int(ring_sample)
                CONFIG["max_rings"] = int(max_rings)
                with st.spinner("Detecting rings..."):
                    start = time.time()
                    graph = GraphAnalytics()
                    ring_out, ring_df = graph.model2_ring_detection(st.session_state.graph_feature_df)
                    st.session_state.step_times["Rings"] = time.time() - start
                    st.session_state.graph_feature_df = ring_out
                    st.session_state.ring_df = ring_df
                    st.session_state.current_step = max(st.session_state.current_step, 4)
                st.rerun()

    if st.session_state.graph_features is not None:
        gf = st.session_state.graph_features

        if isinstance(gf, pd.DataFrame) and len(gf) > 0:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                render_kpi("scatter_plot", "Nodes", f"{len(gf):,}", "", "orange")
            with col2:
                if "graph_community_id" in gf.columns:
                    render_kpi("groups", "Communities", f"{gf['graph_community_id'].nunique()}", "", "teal")
            with col3:
                if "graph_cycle_flag" in gf.columns:
                    render_kpi("warning", "Cycle Nodes", f"{gf['graph_cycle_flag'].sum():,}", "", "red")
            with col4:
                ring_count = len(st.session_state.ring_df) if st.session_state.ring_df is not None and isinstance(st.session_state.ring_df, pd.DataFrame) else 0
                render_kpi("toll", "Rings Found", f"{ring_count}", "", "rose")

    if st.session_state.graph_feature_df is not None:
        render_section("Network Visualization", "share")
        render_section_note("This overview shows how customers, accounts, and counterparties connect across the sampled network.")
        render_network_graph(st.session_state.graph_feature_df, st.session_state.ring_df)

        if st.session_state.graph_features is not None and isinstance(st.session_state.graph_features, pd.DataFrame) and len(st.session_state.graph_features) > 0:
            render_section("Interactive Network Explorer", "travel_explore")
            render_section_note("Focus a node to change the neighborhood depth, node cap, and color encoding.")
            gf = st.session_state.graph_features.copy()
            top_nodes = gf.sort_values(
                ["graph_pagerank", "graph_degree_centrality"],
                ascending=[False, False],
            )["node_id"].head(120).tolist()
            focus_col1, focus_col2, focus_col3 = st.columns([1.4, 0.7, 0.7])
            with focus_col1:
                focus_node = st.selectbox("Focus node", ["All nodes"] + top_nodes)
            with focus_col2:
                depth = st.slider("Neighborhood depth", 1, 2, 1)
            with focus_col3:
                node_cap = st.slider("Node cap", 30, 150, 80, step=10)
            color_by = st.selectbox("Color by", ["Node Type", "Community", "Risk"], index=0)
            selected_focus = None if focus_node == "All nodes" else focus_node
            explorer_fig = build_interactive_network_figure(
                st.session_state.graph_feature_df,
                gf,
                st.session_state.ring_df,
                focus_node=selected_focus,
                depth=depth,
                max_nodes=node_cap,
                color_by=color_by,
            )
            st.plotly_chart(
                explorer_fig,
                use_container_width=True,
                config={"displayModeBar": True, "scrollZoom": True, "responsive": True},
            )

            metric_cols = st.columns(4)
            with metric_cols[0]:
                render_kpi("hub", "Top Pagerank", f"{gf['graph_pagerank'].max():.4f}" if "graph_pagerank" in gf.columns else "Pending", "", "orange")
            with metric_cols[1]:
                render_kpi("share", "Avg Component", f"{gf['graph_component_size'].mean():.1f}" if "graph_component_size" in gf.columns else "Pending", "", "teal")
            with metric_cols[2]:
                render_kpi("speed", "Core Max", f"{int(gf['graph_core_number'].max())}" if "graph_core_number" in gf.columns else "Pending", "", "gold")
            with metric_cols[3]:
                ring_nodes = 0
                if st.session_state.ring_df is not None and isinstance(st.session_state.ring_df, pd.DataFrame) and len(st.session_state.ring_df) > 0 and "ring_members" in st.session_state.ring_df.columns:
                    ring_nodes = len(
                        {
                            str(node)
                            for members in st.session_state.ring_df["ring_members"]
                            for node in (members if isinstance(members, (list, tuple, set)) else str(members).split(","))
                        }
                    )
                render_kpi("warning", "Ring Nodes", f"{ring_nodes:,}" if ring_nodes else "Pending", "", "red")

    if st.session_state.ring_df is not None and isinstance(st.session_state.ring_df, pd.DataFrame) and len(st.session_state.ring_df) > 0:
        render_section("Detected Rings", "toll")
        render_section_note("Filter ring candidates by size and risk to inspect the strongest suspicious structures first.")

        ring_df = st.session_state.ring_df
        ring_controls = st.columns([1.0, 1.0, 1.0])
        with ring_controls[0]:
            min_members = st.slider(
                "Minimum members",
                3,
                int(ring_df["ring_member_count"].max()) if "ring_member_count" in ring_df.columns else 3,
                3,
            )
        with ring_controls[1]:
            min_risk = st.slider("Minimum risk", 0.0, 1.0, 0.35, 0.05)
        with ring_controls[2]:
            ring_kind = st.multiselect(
                "Ring type",
                options=sorted(ring_df["ring_type"].astype(str).unique().tolist()) if "ring_type" in ring_df.columns else ["cycle"],
                default=sorted(ring_df["ring_type"].astype(str).unique().tolist()) if "ring_type" in ring_df.columns else ["cycle"],
            )

        ring_view = ring_df.copy()
        if "ring_member_count" in ring_view.columns:
            ring_view = ring_view[ring_view["ring_member_count"] >= min_members]
        if "ring_risk_score" in ring_view.columns:
            ring_view = ring_view[ring_view["ring_risk_score"] >= min_risk]
        if ring_kind and "ring_type" in ring_view.columns:
            ring_view = ring_view[ring_view["ring_type"].astype(str).isin(ring_kind)]

        if ring_view.empty:
            st.info("No rings match the current filters.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                if "ring_member_count" in ring_view.columns and "ring_risk_score" in ring_view.columns:
                    fig = px.scatter(
                        ring_view,
                        x="ring_member_count",
                        y="ring_risk_score",
                        size="ring_total_amount" if "ring_total_amount" in ring_view.columns else None,
                        color="ring_type" if "ring_type" in ring_view.columns else None,
                        hover_data=[c for c in ["ring_id", "ring_edge_count", "ring_avg_edge_weight", "ring_path_signature"] if c in ring_view.columns],
                        title="Ring size versus risk",
                    )
                    fig = apply_dark_theme(fig)
                    fig.update_layout(height=380)
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                if "ring_risk_score" in ring_view.columns:
                    fig = px.histogram(
                        ring_view,
                        x="ring_risk_score",
                        nbins=20,
                        color_discrete_sequence=[PWC_COLORS["primary"]],
                        title="Ring Risk Score Distribution",
                    )
                    fig = apply_dark_theme(fig)
                    fig.update_layout(height=380)
                    st.plotly_chart(fig, use_container_width=True)

            inspect_ring = st.selectbox("Inspect ring", ring_view["ring_id"].astype(str).tolist())
            selected_ring = ring_view[ring_view["ring_id"].astype(str) == str(inspect_ring)].iloc[0]
            detail_cols = st.columns(4)
            with detail_cols[0]:
                render_kpi("groups", "Members", f"{int(selected_ring.get('ring_member_count', 0)):,}", "", "orange")
            with detail_cols[1]:
                render_kpi("share", "Edges", f"{int(selected_ring.get('ring_edge_count', 0)):,}", "", "teal")
            with detail_cols[2]:
                render_kpi("warning", "Risk", f"{float(selected_ring.get('ring_risk_score', 0.0)):.4f}", "", "red")
            with detail_cols[3]:
                render_kpi("payments", "Amount", f"{float(selected_ring.get('ring_total_amount', 0.0)):.0f}", "", "gold")

            st.code(str(selected_ring.get("ring_path_signature", "")))
            members = selected_ring.get("ring_members", [])
            if isinstance(members, str):
                members = [item.strip() for item in members.split(",") if item.strip()]
            st.caption(", ".join(map(str, members[:20])))
            st.dataframe(ring_view.head(100), use_container_width=True, height=300)


def page_model_training():
    render_top_bar("Model Training Studio", "Configure, train, and evaluate classification models")
    render_progress_steps(5)
    render_page_intro("Choose the champion and challenger, define the split window, and compare their class performance and diagnostics before packaging the model.")

    working_df = st.session_state.graph_feature_df if st.session_state.graph_feature_df is not None else st.session_state.feature_df

    if working_df is None:
        st.warning("Complete previous steps first")
        return

    render_section("Algorithm Selection", "model_training")
    render_section_note("Pick the model family and training controls first so the champion/challenger comparison stays consistent.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="data-card"><h3>Champion Model</h3>', unsafe_allow_html=True)
        base_model_options = ["Random Forest", "Gradient Boosting", "Extra Trees"]
        champion_index = base_model_options.index(CONFIG.get("classifier_model", "Random Forest")) if CONFIG.get("classifier_model", "Random Forest") in base_model_options else 0
        base_model = st.selectbox("Classifier", base_model_options, index=champion_index)
        n_estimators = st.slider("Estimators", 50, 1000, int(CONFIG.get("classifier_estimators", 220)), step=50)
        max_depth = st.slider("Max Depth", 3, 30, int(CONFIG.get("classifier_max_depth", 12)))
        min_leaf = st.slider("Min Samples Leaf", 1, 20, int(CONFIG.get("classifier_min_samples_leaf", 4)))
        calibration_options = ["sigmoid", "isotonic"]
        calibration_index = calibration_options.index(CONFIG.get("calibration_method", "sigmoid")) if CONFIG.get("calibration_method", "sigmoid") in calibration_options else 0
        calibration = st.selectbox("Calibration", calibration_options, index=calibration_index)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="data-card"><h3>Challenger Model</h3>', unsafe_allow_html=True)
        challenger_options = ["Logistic Regression", "SVM Linear", "Naive Bayes"]
        challenger_index = challenger_options.index(CONFIG.get("challenger_model", "Logistic Regression")) if CONFIG.get("challenger_model", "Logistic Regression") in challenger_options else 0
        challenger = st.selectbox("Challenger Classifier", challenger_options, index=challenger_index)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="data-card"><h3>Split Configuration</h3>', unsafe_allow_html=True)
        train_end = st.date_input("Train End", value=pd.to_datetime(CONFIG["train_end"]))
        valid_end = st.date_input("Valid End", value=pd.to_datetime(CONFIG["valid_end"]))
        st.markdown('</div>', unsafe_allow_html=True)

    CONFIG["classifier_model"] = base_model
    CONFIG["classifier_estimators"] = int(n_estimators)
    CONFIG["classifier_max_depth"] = int(max_depth)
    CONFIG["classifier_min_samples_leaf"] = int(min_leaf)
    CONFIG["calibration_method"] = calibration
    CONFIG["challenger_model"] = challenger
    CONFIG["train_end"] = str(train_end)
    CONFIG["valid_end"] = str(valid_end)

    render_section("Training Pipeline", "play_arrow")
    render_section_note("Run each stage separately when you want to inspect timing, or use Run All for a complete pass.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        run_split = st.button("Split Data", type="primary", use_container_width=True)
    with col2:
        run_seq = st.button("Sequence Models", type="primary", use_container_width=True)
    with col3:
        run_cls = st.button("Train Classifier", type="primary", use_container_width=True)
    with col4:
        run_all = st.button("Run All", type="primary", use_container_width=True)

    if run_split or run_all:
        from multiclass_model import MulticlassModel
        modeler = MulticlassModel()

        with st.spinner("Splitting..."):
            start = time.time()
            train_df, valid_df, test_df = modeler.split_time_based(working_df, str(train_end), str(valid_end))
            st.session_state.step_times["Split"] = time.time() - start
            st.session_state.train_df = train_df
            st.session_state.valid_df = valid_df
            st.session_state.test_df = test_df

    if run_seq or run_all:
        if st.session_state.train_df is None:
            st.warning("Split data first")
        else:
            from sequence_models import SequenceModels
            seq = SequenceModels()
            train_df, valid_df, test_df = st.session_state.train_df, st.session_state.valid_df, st.session_state.test_df

            with st.spinner("Training sequence models..."):
                start = time.time()
                model3, train_df, valid_df, test_df = seq.model3_hazard(train_df, valid_df, test_df)
                model4, train_df, valid_df, test_df = seq.model4_hmm(train_df, valid_df, test_df)
                model5_outputs = seq.model5_lstm_and_transformer(train_df, valid_df, test_df)
                st.session_state.step_times["Sequence"] = time.time() - start

                st.session_state.model3 = model3
                st.session_state.model4 = model4
                st.session_state.model5_outputs = model5_outputs
                st.session_state.train_df = train_df
                st.session_state.valid_df = valid_df
                st.session_state.test_df = test_df

    if run_cls or run_all:
        if st.session_state.train_df is None:
            st.warning("Split data first")
        else:
            from multiclass_model import MulticlassModel
            modeler = MulticlassModel()

            with st.spinner("Training classifier..."):
                start = time.time()
                result = modeler.model6_multiclass(st.session_state.train_df, st.session_state.valid_df, st.session_state.test_df)
                st.session_state.step_times["Classifier"] = time.time() - start

                artifacts, fcols, yv, yt, vp, tp, vpred, tpred, vpch, tpch, fi = result
                st.session_state.model6_artifacts = artifacts
                st.session_state.feature_cols = fcols
                st.session_state.y_test = yt
                st.session_state.test_prob = tp
                st.session_state.test_pred = tpred
                st.session_state.test_pred_ch = tpch
                st.session_state.feature_importance = fi
                st.session_state.current_step = max(st.session_state.current_step, 5)
            st.rerun()

    if st.session_state.train_df is not None:
        col1, col2, col3 = st.columns(3)
        with col1:
            render_kpi("school", "Training", f"{len(st.session_state.train_df):,}", "", "green")
        with col2:
            render_kpi("quiz", "Validation", f"{len(st.session_state.valid_df):,}", "", "gold")
        with col3:
            render_kpi("assignment", "Test", f"{len(st.session_state.test_df):,}", "", "red")

    if st.session_state.model6_artifacts is not None:
        from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

        y_test = st.session_state.y_test
        test_pred = st.session_state.test_pred
        test_pred_ch = st.session_state.test_pred_ch
        le = st.session_state.model6_artifacts.label_encoder

        render_section("Model Performance", "leaderboard")
        render_section_note("Use the report, confusion matrix, and feature importance views to compare rank order and class separation.")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            render_kpi("emoji_events", "Champ Accuracy", f"{accuracy_score(y_test, test_pred):.4f}", "", "green")
        with col2:
            render_kpi("star", "Champ Macro F1", f"{f1_score(y_test, test_pred, average='macro'):.4f}", "", "green")
        with col3:
            render_kpi("psychology", "Chall Accuracy", f"{accuracy_score(y_test, test_pred_ch):.4f}", "", "gold")
        with col4:
            render_kpi("grade", "Chall Macro F1", f"{f1_score(y_test, test_pred_ch, average='macro'):.4f}", "", "gold")

        tab1, tab2, tab3 = st.tabs(["Classification Report", "Confusion Matrix", "Feature Importance"])

        with tab1:
            st.caption("Per-class precision, recall, and F1 scores for the champion model on the held-out test set.")
            report = classification_report(y_test, test_pred, target_names=le.classes_, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True, height=400)

        with tab2:
            st.caption("The confusion matrix highlights where mule typologies are being confused with one another.")
            cm = confusion_matrix(y_test, test_pred)
            cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
            fig = px.imshow(cm_df, text_auto=True, color_continuous_scale=[[0, "#F8ECE1"], [0.5, "#D04A02"], [1, "#FFB600"]])
            fig = apply_dark_theme(fig)
            fig.update_layout(height=600, title="Confusion Matrix")
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.caption("Feature importance shows which engineered signals contribute most to the multiclass decisioning.")
            fi = st.session_state.feature_importance
            top_n = st.slider("Top N", 10, 50, 25)
            top_fi = fi.head(top_n)
            fig = px.bar(top_fi, y="feature", x="importance", orientation="h",
                         color="importance",
                         color_continuous_scale=[[0, "#F7E6DA"], [0.5, "#D04A02"], [1, "#FFB600"]])
            fig = apply_dark_theme(fig)
            fig.update_layout(height=max(400, top_n * 22), showlegend=False, title=f"Top {top_n} Features",
                              yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)

        render_section("Model Diagnostics", "analytics")
        render_section_note("The diagnostics surface confidence, mean class probabilities, and low-confidence test cases.")
        diag_cols = st.columns([1.0, 1.0, 1.0])
        with diag_cols[0]:
            if st.session_state.test_prob is not None:
                confidence = np.max(st.session_state.test_prob, axis=1)
                fig = px.histogram(
                    pd.DataFrame({"Confidence": confidence}),
                    x="Confidence",
                    nbins=30,
                    title="Prediction confidence",
                )
                fig = apply_dark_theme(fig)
                fig.update_layout(height=320)
                st.plotly_chart(fig, use_container_width=True)
        with diag_cols[1]:
            if st.session_state.test_prob is not None and len(st.session_state.test_prob) > 0:
                avg_prob = pd.DataFrame(st.session_state.test_prob, columns=le.classes_).mean().sort_values(ascending=False).reset_index()
                avg_prob.columns = ["Class", "Mean Probability"]
                fig = px.bar(avg_prob, x="Class", y="Mean Probability", color="Mean Probability", title="Mean predicted probability")
                fig = apply_dark_theme(fig)
                fig.update_layout(height=320, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        with diag_cols[2]:
            if st.session_state.test_prob is not None:
                uncertain_idx = np.argsort(np.max(st.session_state.test_prob, axis=1))[:20]
                uncertain = pd.DataFrame({
                    "actual": le.inverse_transform(y_test[uncertain_idx]),
                    "predicted": le.inverse_transform(test_pred[uncertain_idx]),
                    "confidence": np.max(st.session_state.test_prob[uncertain_idx], axis=1),
                }).sort_values("confidence")
                st.dataframe(uncertain.round(4), use_container_width=True, height=320)

        if st.session_state.test_prob is not None:
            per_class = []
            report = classification_report(y_test, test_pred, target_names=le.classes_, output_dict=True)
            for cls in le.classes_:
                if cls in report:
                    per_class.append({
                        "Class": cls,
                        "Precision": report[cls]["precision"],
                        "Recall": report[cls]["recall"],
                        "F1": report[cls]["f1-score"],
                    })
            if per_class:
                per_class_df = pd.DataFrame(per_class)
                fig = px.bar(
                    per_class_df.melt(id_vars="Class", var_name="Metric", value_name="Score"),
                    x="Class",
                    y="Score",
                    color="Metric",
                    barmode="group",
                    title="Per-class metrics",
                )
                fig = apply_dark_theme(fig)
                fig.update_layout(height=420)
                st.plotly_chart(fig, use_container_width=True)


def page_model_inference():
    render_top_bar("Model Inference", "Generate fresh synthetic test data, compare trained models, and explain single predictions")
    render_page_intro("This page regenerates an out-of-sample synthetic batch, scores it with the trained champion model, and lets you inspect the HMM sequence layer for abnormal behavior.")

    bundle = get_active_model_bundle()

    render_section("Inference Controls", "science")
    render_section_note("Generate a fresh test batch using the current run profile. The same entity resolution, feature engineering, graph enrichment, and sequence steps are applied before scoring. For the quickest demo loop, switch the sidebar run profile to Fast before generating the batch.")

    control_cols = st.columns([0.9, 1.0, 1.2])
    with control_cols[0]:
        default_seed = int(st.session_state.inference_batch["seed"]) if st.session_state.inference_batch else int(CONFIG["random_state"]) + 17
        inference_seed = st.number_input("Inference seed", value=default_seed, min_value=1, max_value=99999)
    with control_cols[1]:
        model_choice = st.selectbox(
            "Inference model",
            ["Random Forest Champion", "Hidden Markov Sequence"],
            index=0,
        )
    with control_cols[2]:
        st.caption(f"Using run profile: {st.session_state.demo_profile}")
        model_source = bundle["source"] if bundle is not None else "No trained champion available yet"
        st.caption(f"Champion source: {model_source}")

    if st.button("Generate Fresh Test Batch", type="primary", use_container_width=True):
        with st.spinner("Generating fresh synthetic test batch and enrichment layers..."):
            st.session_state.inference_batch = generate_fresh_inference_batch(int(inference_seed))
        st.rerun()

    batch = st.session_state.inference_batch
    if batch is None:
        st.info("Generate a fresh test batch to open the inference workspace.")
        return

    test_df = batch.get("test_df")
    if test_df is None or len(test_df) == 0:
        st.warning("The generated test batch did not contain a usable holdout window. Try another seed or a larger profile.")
        return

    render_section("Fresh Batch Snapshot", "insights")
    render_section_note("These metrics describe the generated holdout batch that is being used for inference and evaluation.")

    risky_count = int(test_df["label"].isin(CONFIG["risky_labels"]).sum()) if "label" in test_df.columns else 0
    runtime_secs = sum(batch.get("timings", {}).values())
    snapshot_cols = st.columns(4)
    with snapshot_cols[0]:
        render_kpi("data_array", "Test Records", f"{len(test_df):,}", "Holdout rows scored in this view", "orange")
    with snapshot_cols[1]:
        render_kpi("warning", "Risky Labels", f"{risky_count:,}", "Rows belonging to risky mule classes", "red")
    with snapshot_cols[2]:
        render_kpi("hub", "Rings", f"{int(batch.get('ring_count', 0)):,}", "Detected during fresh graph enrichment", "teal")
    with snapshot_cols[3]:
        render_kpi("timer", "Build Time", f"{runtime_secs:.1f}s", batch.get("generated_at", ""), "gold")

    timing_df = pd.DataFrame(
        [{"Step": key, "Seconds": value} for key, value in batch.get("timings", {}).items()]
    )
    if not timing_df.empty:
        fig = px.bar(
            timing_df,
            x="Seconds",
            y="Step",
            orientation="h",
            color="Seconds",
            color_continuous_scale=[[0, "#F7E6DA"], [0.5, "#D04A02"], [1, "#FFB600"]],
            title="Fresh batch preparation timing",
        )
        fig = apply_dark_theme(fig)
        fig.update_layout(height=max(320, len(timing_df) * 26), showlegend=False, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    if model_choice == "Random Forest Champion":
        render_section("Champion Multiclass Output", "model_training")
        render_section_note("This view uses the calibrated champion classifier to predict mule typologies for the fresh holdout batch and explains one selected prediction with LIME.")

        if bundle is None:
            st.warning("Train the champion model in Step 5 or save the export bundle before using multiclass inference.")
            return

        inference_result = run_multiclass_inference(bundle, test_df)
        scored_df = inference_result["scored_df"]

        metric_cols = st.columns(4)
        with metric_cols[0]:
            render_kpi("inventory", "Model", bundle["model_label"], bundle["source"], "orange")
        with metric_cols[1]:
            render_kpi("precision_manufacturing", "Predicted Classes", f"{scored_df['predicted_label'].nunique():,}", "Unique classes predicted on this batch", "teal")
        with metric_cols[2]:
            avg_conf = float(scored_df["prediction_confidence"].mean()) if len(scored_df) else 0.0
            render_kpi("verified", "Avg Confidence", f"{avg_conf:.3f}", "Mean top-class probability", "gold")
        with metric_cols[3]:
            if inference_result["macro_f1"] is not None:
                render_kpi("leaderboard", "Macro F1", f"{inference_result['macro_f1']:.3f}", "Fresh synthetic holdout", "green")
            else:
                render_kpi("leaderboard", "Macro F1", "Unavailable", "Ground-truth labels were not aligned", "green")

        rf_tabs = st.tabs(["Predictions", "Confusion Matrix", "Classification Report", "LIME Explanation"])

        with rf_tabs[0]:
            st.caption("Review the predicted class mix, confidence spread, and the highest-confidence scored rows.")
            pred_cols = st.columns(2)
            with pred_cols[0]:
                pred_dist = scored_df["predicted_label"].value_counts().reset_index()
                pred_dist.columns = ["Predicted Label", "Count"]
                fig = px.bar(pred_dist, x="Predicted Label", y="Count", color="Predicted Label", title="Predicted class distribution")
                fig = apply_dark_theme(fig)
                fig.update_layout(height=360, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            with pred_cols[1]:
                fig = px.histogram(scored_df, x="prediction_confidence", nbins=30, color_discrete_sequence=[PWC_COLORS["primary"]], title="Prediction confidence")
                fig = apply_dark_theme(fig)
                fig.update_layout(height=360)
                st.plotly_chart(fig, use_container_width=True)

            preview_cols = [c for c in ["customer_id", "account_id", "channel", "label", "predicted_label", "prediction_confidence", "top_predictions"] if c in scored_df.columns]
            st.dataframe(
                scored_df.sort_values("prediction_confidence", ascending=False)[preview_cols].head(150),
                use_container_width=True,
                height=360,
            )

            if "is_correct" in scored_df.columns:
                errors = scored_df[~scored_df["is_correct"]].copy()
                if not errors.empty:
                    st.caption("Top confident misclassifications")
                    error_cols = [c for c in ["customer_id", "account_id", "label", "predicted_label", "prediction_confidence", "top_predictions"] if c in errors.columns]
                    st.dataframe(
                        errors.sort_values("prediction_confidence", ascending=False)[error_cols].head(50),
                        use_container_width=True,
                        height=240,
                    )

        with rf_tabs[1]:
            st.caption("The confusion matrix shows where the champion model overlaps across mule typologies on the fresh holdout batch.")
            if inference_result["confusion_df"] is not None:
                fig = px.imshow(
                    inference_result["confusion_df"],
                    text_auto=True,
                    color_continuous_scale=[[0, "#F8ECE1"], [0.5, "#D04A02"], [1, "#FFB600"]],
                    title="Fresh holdout confusion matrix",
                )
                fig = apply_dark_theme(fig)
                fig.update_layout(height=700)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Ground-truth labels were not available for a full confusion matrix.")

        with rf_tabs[2]:
            st.caption("The classification report summarizes precision, recall, and F1 for each mule typology.")
            if inference_result["report_df"] is not None:
                st.dataframe(inference_result["report_df"].round(4), use_container_width=True, height=500)
            else:
                st.info("Ground-truth labels were not available for a full classification report.")

        with rf_tabs[3]:
            st.caption("LIME explains one scored record in the processed model feature space. This is the most reliable local explanation path for the champion model in this app.")
            explain_limit = min(len(scored_df), 120)
            explain_options = []
            explain_map = {}
            for position in range(explain_limit):
                row = scored_df.iloc[position]
                row_label = row.get("customer_id", row.get("account_id", f"Row {position + 1}"))
                explain_text = f"{position + 1}. {row_label} | pred={row['predicted_label']} | conf={row['prediction_confidence']:.2f}"
                if "label" in row.index:
                    explain_text += f" | actual={row['label']}"
                explain_options.append(explain_text)
                explain_map[explain_text] = position

            if not explain_options:
                st.info("No scored rows are available for explanation.")
            else:
                explain_cols = st.columns([1.4, 0.7])
                with explain_cols[0]:
                    selected_option = st.selectbox("Record to explain", explain_options, key="lime_record_select")
                with explain_cols[1]:
                    lime_feature_count = st.slider("Top features", 6, 15, 10, key="lime_feature_count")

                if st.button("Generate LIME Explanation", type="secondary", key="lime_generate_button"):
                    selected_position = explain_map[selected_option]
                    try:
                        with st.spinner("Generating local explanation..."):
                            lime_df = compute_lime_explanation(
                                inference_result["processed"],
                                bundle,
                                selected_position,
                                inference_result["pred_idx"][selected_position],
                                num_features=lime_feature_count,
                            )
                    except RuntimeError as exc:
                        st.warning(str(exc))
                    else:
                        if lime_df.empty:
                            st.info("LIME did not return a usable explanation for this record.")
                        else:
                            explanation_fig = px.bar(
                                lime_df.sort_values("weight"),
                                x="weight",
                                y="feature",
                                orientation="h",
                                color="direction",
                                color_discrete_map={
                                    "Supports prediction": PWC_COLORS["primary"],
                                    "Pushes against prediction": PWC_COLORS["accent_teal"],
                                },
                                title="Local feature contributions",
                            )
                            explanation_fig = apply_dark_theme(explanation_fig)
                            explanation_fig.update_layout(height=max(360, len(lime_df) * 34))
                            st.plotly_chart(explanation_fig, use_container_width=True)
                            st.dataframe(lime_df.round(4), use_container_width=True, height=280)

    else:
        render_section("Hidden Markov Sequence View", "timeline")
        render_section_note("The HMM is an unsupervised sequence layer. It does not directly assign mule typologies, but it surfaces unusual state transitions and abnormal behavior intensity.")

        hmm_df = test_df.copy()
        if "hmm_sequence_anomaly_score" not in hmm_df.columns:
            st.warning("HMM outputs were not available on this batch. Generate another batch or verify the sequence stage.")
            return

        anomaly_min = float(hmm_df["hmm_sequence_anomaly_score"].min())
        anomaly_max = float(hmm_df["hmm_sequence_anomaly_score"].max())
        default_threshold = float(hmm_df["hmm_sequence_anomaly_score"].quantile(0.85))
        if anomaly_max > anomaly_min:
            anomaly_threshold = st.slider(
                "Anomaly threshold",
                min_value=anomaly_min,
                max_value=anomaly_max,
                value=default_threshold,
                step=max((anomaly_max - anomaly_min) / 100, 0.001),
            )
        else:
            anomaly_threshold = anomaly_max
            st.caption("All HMM anomaly scores are currently identical on this batch, so the threshold is fixed.")

        hmm_df["predicted_risky_flag"] = (hmm_df["hmm_sequence_anomaly_score"] >= anomaly_threshold).astype(int)
        if "label" in hmm_df.columns:
            hmm_df["actual_risky_flag"] = hmm_df["label"].isin(CONFIG["risky_labels"]).astype(int)

        hmm_metric_cols = st.columns(4)
        with hmm_metric_cols[0]:
            render_kpi("timeline", "Rows Scored", f"{len(hmm_df):,}", "Sequence rows in the holdout batch", "orange")
        with hmm_metric_cols[1]:
            render_kpi("psychology", "Avg Anomaly", f"{hmm_df['hmm_sequence_anomaly_score'].mean():.3f}", "Mean HMM anomaly intensity", "teal")
        with hmm_metric_cols[2]:
            render_kpi("warning", "Flagged Rows", f"{int(hmm_df['predicted_risky_flag'].sum()):,}", "Rows above the anomaly threshold", "red")
        with hmm_metric_cols[3]:
            state_count = hmm_df["hmm_state"].nunique() if "hmm_state" in hmm_df.columns else 0
            render_kpi("hub", "States", f"{state_count}", f"Sequence model: {batch.get('sequence_model_type', 'unknown')}", "gold")

        hmm_tabs = st.tabs(["Sequence Risk", "State Overlay", "Binary Confusion", "Flagged Cases"])

        with hmm_tabs[0]:
            st.caption("Review anomaly spread and how the sequence risk layer varies across actual labels.")
            seq_cols = st.columns(2)
            with seq_cols[0]:
                fig = px.histogram(
                    hmm_df,
                    x="hmm_sequence_anomaly_score",
                    nbins=30,
                    color_discrete_sequence=[PWC_COLORS["primary"]],
                    title="HMM anomaly score distribution",
                )
                fig = apply_dark_theme(fig)
                fig.update_layout(height=360)
                st.plotly_chart(fig, use_container_width=True)
            with seq_cols[1]:
                if "label" in hmm_df.columns:
                    sample_size = min(len(hmm_df), 2000)
                    plot_df = hmm_df.sample(sample_size, random_state=CONFIG["random_state"]) if len(hmm_df) > sample_size else hmm_df
                    fig = px.box(
                        plot_df,
                        x="label",
                        y="hmm_sequence_anomaly_score",
                        color="label",
                        title="Anomaly score by actual label",
                    )
                    fig = apply_dark_theme(fig)
                    fig.update_layout(height=360, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

        with hmm_tabs[1]:
            st.caption("This overlay shows how actual labels are distributed across HMM states. It is a state interpretation aid, not a direct multiclass HMM prediction.")
            if {"hmm_state", "label"}.issubset(hmm_df.columns):
                state_overlay = pd.crosstab(hmm_df["hmm_state"], hmm_df["label"])
                fig = px.imshow(
                    state_overlay,
                    text_auto=True,
                    color_continuous_scale=[[0, "#F8ECE1"], [0.5, "#D04A02"], [1, "#FFB600"]],
                    title="Actual label distribution by HMM state",
                )
                fig = apply_dark_theme(fig)
                fig.update_layout(height=520)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("State overlay becomes available when both HMM states and actual labels are present.")

        with hmm_tabs[2]:
            st.caption("Because the HMM is unsupervised, the direct evaluation here is binary: flagged sequence versus actual risky label.")
            if "actual_risky_flag" in hmm_df.columns:
                from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score

                binary_cm = pd.DataFrame(
                    confusion_matrix(hmm_df["actual_risky_flag"], hmm_df["predicted_risky_flag"]),
                    index=["Actual Legit", "Actual Risky"],
                    columns=["Pred Legit", "Pred Risky"],
                )
                fig = px.imshow(
                    binary_cm,
                    text_auto=True,
                    color_continuous_scale=[[0, "#F8ECE1"], [0.5, "#D04A02"], [1, "#FFB600"]],
                    title="Binary risky-versus-legit confusion matrix",
                )
                fig = apply_dark_theme(fig)
                fig.update_layout(height=420)
                st.plotly_chart(fig, use_container_width=True)

                binary_metrics = pd.DataFrame(
                    [
                        {"Metric": "Precision", "Score": precision_score(hmm_df["actual_risky_flag"], hmm_df["predicted_risky_flag"], zero_division=0)},
                        {"Metric": "Recall", "Score": recall_score(hmm_df["actual_risky_flag"], hmm_df["predicted_risky_flag"], zero_division=0)},
                        {"Metric": "F1", "Score": f1_score(hmm_df["actual_risky_flag"], hmm_df["predicted_risky_flag"], zero_division=0)},
                    ]
                )
                st.dataframe(binary_metrics.round(4), use_container_width=True, height=160)

                binary_report = pd.DataFrame(
                    classification_report(
                        hmm_df["actual_risky_flag"],
                        hmm_df["predicted_risky_flag"],
                        target_names=["legit", "risky"],
                        output_dict=True,
                        zero_division=0,
                    )
                ).transpose()
                st.dataframe(binary_report.round(4), use_container_width=True, height=240)
            else:
                st.info("Actual labels were not available for a binary risky-versus-legit evaluation.")

        with hmm_tabs[3]:
            st.caption("These are the highest-scoring anomalous sequences according to the HMM layer.")
            flagged_cols = [c for c in ["customer_id", "account_id", "channel", "label", "hmm_state", "hmm_sequence_anomaly_score", "hazard_score"] if c in hmm_df.columns]
            st.dataframe(
                hmm_df.sort_values("hmm_sequence_anomaly_score", ascending=False)[flagged_cols].head(150),
                use_container_width=True,
                height=420,
            )


def page_alerts():
    render_top_bar("Alert Engine", "Decision engine and intelligent alert packaging")
    render_progress_steps(6)
    render_page_intro("Score the held-out data, package the highest-risk cases, and inspect the reasons that pushed each alert into the queue.")

    if st.session_state.model6_artifacts is None:
        st.warning("Train models first in Step 5")
        return

    render_section("Decision and Packaging", "gavel")
    render_section_note("Run the decision engine first, then convert the scored test set into alert packets and triage-ready thresholds.")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Run Decision Engine", type="primary", use_container_width=True):
            from alert_engine import AlertEngine

            with st.spinner("Running decision engine..."):
                start = time.time()
                ae = AlertEngine()
                test_df = ae.model7_decision_engine(
                    st.session_state.test_df, st.session_state.test_prob,
                    st.session_state.model6_artifacts, st.session_state.model5_outputs)
                st.session_state.step_times["Decision"] = time.time() - start
                st.session_state.test_df = test_df
            st.rerun()

    with col2:
        if st.button("Generate Alerts", type="primary", use_container_width=True):
            from alert_engine import AlertEngine

            with st.spinner("Generating alerts..."):
                start = time.time()
                ae = AlertEngine()
                result = ae.model8_alert_pack(st.session_state.test_df, st.session_state.test_prob, st.session_state.model6_artifacts)
                st.session_state.step_times["Alerts"] = time.time() - start
                st.session_state.alert_output = result[0]
                st.session_state.threshold_table = result[1]
                st.session_state.channel_thresholds = result[2]
                st.session_state.class_thresholds = result[3]
                st.session_state.threshold_opt = result[5]
                st.session_state.current_step = max(st.session_state.current_step, 6)
            st.rerun()

    if st.session_state.alert_output is not None and isinstance(st.session_state.alert_output, pd.DataFrame):
        ao = st.session_state.alert_output

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            render_kpi("notifications", "Total Alerts", f"{len(ao):,}", "", "orange")
        with col2:
            high = (ao["risk_tier"] == "HIGH").sum() if "risk_tier" in ao.columns else 0
            render_kpi("error", "High Risk", f"{high:,}", "", "red")
        with col3:
            med = (ao["risk_tier"] == "MEDIUM").sum() if "risk_tier" in ao.columns else 0
            render_kpi("warning", "Medium Risk", f"{med:,}", "", "gold")
        with col4:
            low = (ao["risk_tier"] == "LOW").sum() if "risk_tier" in ao.columns else 0
            render_kpi("check_circle", "Low Risk", f"{low:,}", "", "green")

        if "risk_tier" in ao.columns:
            tier_dist = ao["risk_tier"].value_counts().reset_index()
            tier_dist.columns = ["Tier", "Count"]
            colors = {"HIGH": PWC_COLORS["danger"], "MEDIUM": PWC_COLORS["warning"], "LOW": PWC_COLORS["success"]}
            fig = px.bar(tier_dist, x="Tier", y="Count", color="Tier",
                         color_discrete_map=colors)
            fig = apply_dark_theme(fig)
            fig.update_layout(height=350, showlegend=False, title="Alert Distribution by Risk Tier")
            st.plotly_chart(fig, use_container_width=True)

        render_section("Alert Details", "list_alt")
        render_section_note("This table lists the scored cases in priority order with the most important signal columns exposed.")
        st.dataframe(ao.head(500), use_container_width=True, height=400)

        csv = ao.to_csv(index=False)
        st.download_button("Download Alerts", data=csv, file_name="alerts.csv", mime="text/csv")

        render_section("Alert Triage", "filter_alt")
        render_section_note("Filter by tier, score floor, and queue size to narrow the review set to the cases you care about.")
        triage_cols = st.columns([1.0, 1.0, 1.0])
        with triage_cols[0]:
            tier_choices = sorted(ao["risk_tier"].dropna().astype(str).unique().tolist()) if "risk_tier" in ao.columns else []
            tier_filter = st.multiselect("Risk tiers", tier_choices, default=tier_choices)
        with triage_cols[1]:
            score_floor = st.slider("Min final score", 0.0, 1.0, 0.35, 0.01)
        with triage_cols[2]:
            queue_size = st.slider("Queue size", 20, 500, 100, step=20)

        triage_view = ao.copy()
        if tier_filter and "risk_tier" in triage_view.columns:
            triage_view = triage_view[triage_view["risk_tier"].astype(str).isin(tier_filter)]
        score_col = "final_mule_score" if "final_mule_score" in triage_view.columns else ("mule_score" if "mule_score" in triage_view.columns else None)
        if score_col:
            triage_view = triage_view[pd.to_numeric(triage_view[score_col], errors="coerce").fillna(0) >= score_floor]

        if triage_view.empty:
            st.info("No alerts match the current filters.")
        else:
            left, right = st.columns(2)
            with left:
                if score_col:
                    fig = px.histogram(
                        triage_view,
                        x=score_col,
                        color="risk_tier" if "risk_tier" in triage_view.columns else None,
                        nbins=30,
                        title="Alert score distribution",
                    )
                    fig = apply_dark_theme(fig)
                    fig.update_layout(height=360)
                    st.plotly_chart(fig, use_container_width=True)
            with right:
                if {"final_mule_score", "risk_tier"}.issubset(triage_view.columns):
                    fig = px.box(
                        triage_view,
                        x="risk_tier",
                        y="final_mule_score",
                        color="risk_tier",
                        points="outliers",
                        title="Score spread by risk tier",
                    )
                    fig = apply_dark_theme(fig)
                    fig.update_layout(height=360, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

            st.dataframe(triage_view.head(queue_size), use_container_width=True, height=320)

        if "reasons" in ao.columns:
            render_section("Reason Explorer", "psychology")
            render_section_note("The reason explorer shows the most common drivers behind the packaged alert decisions.")
            reason_rows = []
            for _, row in ao.head(500).iterrows():
                raw_reasons = row.get("reasons", [])
                if isinstance(raw_reasons, str):
                    raw_reasons = [item.strip() for item in raw_reasons.split(";") if item.strip()]
                for reason in raw_reasons:
                    reason_rows.append({
                        "reason": reason,
                        "risk_tier": row.get("risk_tier", "UNKNOWN"),
                        "score": float(row.get("final_mule_score", row.get("mule_score", 0.0))),
                    })
            if reason_rows:
                reasons_df = pd.DataFrame(reason_rows)
                reason_filter_cols = st.columns([1.0, 1.0])
                with reason_filter_cols[0]:
                    reason_query = st.text_input("Search reason", value="")
                with reason_filter_cols[1]:
                    reason_tier = st.selectbox("Reason tier", ["All"] + sorted(reasons_df["risk_tier"].astype(str).unique().tolist()))
                filtered_reasons = reasons_df.copy()
                if reason_query:
                    filtered_reasons = filtered_reasons[filtered_reasons["reason"].str.contains(reason_query, case=False, na=False)]
                if reason_tier != "All":
                    filtered_reasons = filtered_reasons[filtered_reasons["risk_tier"].astype(str) == str(reason_tier)]
                if not filtered_reasons.empty:
                    reason_counts = filtered_reasons["reason"].value_counts().head(15).reset_index()
                    reason_counts.columns = ["Reason", "Count"]
                    fig = px.bar(reason_counts, x="Reason", y="Count", title="Most common alert reasons")
                    fig = apply_dark_theme(fig)
                    fig.update_layout(height=360, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(filtered_reasons.head(150), use_container_width=True, height=260)
                else:
                    st.info("No reasons match the current filters.")


def page_feedback():
    render_top_bar("Feedback Loop", "Weak supervision and model performance feedback")
    render_progress_steps(7)
    render_page_intro("Combine weak supervision and analyst-style review signals so the next iteration has a clearer path to better case quality.")

    if st.session_state.alert_output is None:
        st.warning("Generate alerts first in Step 6")
        return

    render_section("Feedback Run", "loop")
    render_section_note("This pass joins the alert output with feedback signals and prepares the downstream review artifacts.")
    if st.button("Run Feedback Loop", type="primary"):
        from feedback_loop import FeedbackLoop

        working_df = st.session_state.graph_feature_df if st.session_state.graph_feature_df is not None else st.session_state.feature_df

        with st.spinner("Running feedback loop..."):
            start = time.time()
            fb = FeedbackLoop()
            outputs = fb.weak_supervision_and_feedback(working_df, st.session_state.alert_output)
            st.session_state.step_times["Feedback"] = time.time() - start
            st.session_state.feedback_outputs = outputs
            st.session_state.current_step = max(st.session_state.current_step, 7)
        st.rerun()

    if st.session_state.feedback_outputs is not None:
        render_section("Feedback Results", "insights")
        render_section_note("Each table below is a derived artifact from the feedback pass and can be inspected independently.")
        outputs = st.session_state.feedback_outputs
        if isinstance(outputs, dict):
            for key, val in outputs.items():
                if isinstance(val, pd.DataFrame):
                    st.markdown(f'<div class="section-title">{key}</div>', unsafe_allow_html=True)
                    st.dataframe(val.head(100), use_container_width=True, height=250)


def page_export():
    render_top_bar("Save and Export", "Save models, convert to ONNX, and validate with fresh data")
    render_progress_steps(8)
    render_page_intro("Persist the trained artifacts, check the ONNX path, and validate the package against fresh data before handing it off.")

    tab1, tab2, tab3 = st.tabs(["Save Models", "ONNX Conversion", "Fresh Data Test"])

    with tab1:
        render_section("Save Model Artifacts", "save")
        render_section_note("Save the fitted objects and configuration bundle so the model can be reloaded without retraining.")

        if st.session_state.model6_artifacts is None:
            st.warning("Train models first")
        else:
            if st.button("Save All Models", type="primary"):
                artifacts = st.session_state.model6_artifacts
                feature_cols = st.session_state.feature_cols
                from config import CONFIG

                save_dict = {
                    "preprocessor": artifacts.preprocessor,
                    "label_encoder": artifacts.label_encoder,
                    "base_model": artifacts.base_model,
                    "calibrated_model": artifacts.calibrated_model,
                    "challenger_model": artifacts.challenger_model,
                    "feature_names": artifacts.feature_names,
                    "feature_cols": feature_cols,
                    "config": CONFIG,
                    "model_type": "CalibratedClassifierCV(RandomForest)",
                    "n_classes": len(artifacts.label_encoder.classes_),
                    "class_names": list(artifacts.label_encoder.classes_),
                    "n_features_raw": len(feature_cols),
                    "n_features_processed": artifacts.base_model.n_features_in_,
                }

                with open("best_model.pkl", "wb") as f:
                    pickle.dump(save_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

                base_save = {
                    "model": artifacts.base_model,
                    "preprocessor": artifacts.preprocessor,
                    "label_encoder": artifacts.label_encoder,
                    "feature_names": artifacts.feature_names,
                    "feature_cols": feature_cols,
                    "n_features_processed": artifacts.base_model.n_features_in_,
                    "class_names": list(artifacts.label_encoder.classes_),
                }

                with open("base_model_for_onnx.pkl", "wb") as f:
                    pickle.dump(base_save, f, protocol=pickle.HIGHEST_PROTOCOL)

                st.session_state.current_step = max(st.session_state.current_step, 8)
                st.success("Models saved successfully")

            col1, col2 = st.columns(2)
            with col1:
                if os.path.exists("best_model.pkl"):
                    s = os.path.getsize("best_model.pkl") / (1024 * 1024)
                    render_kpi("inventory", "best_model.pkl", f"{s:.1f} MB", "", "orange")
            with col2:
                if os.path.exists("base_model_for_onnx.pkl"):
                    s = os.path.getsize("base_model_for_onnx.pkl") / (1024 * 1024)
                    render_kpi("inventory_2", "base_model_for_onnx.pkl", f"{s:.1f} MB", "", "teal")

    with tab2:
        render_section("ONNX Conversion", "transform")
        render_section_note("This tab converts the packaged classifier into an ONNX-friendly flow for lightweight deployment checks.")

        USE_ONNX = False
        try:
            import onnx
            import onnxruntime as ort
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
            USE_ONNX = True
        except ImportError:
            pass

        if not USE_ONNX:
            st.error("Install ONNX packages: pip install skl2onnx onnx onnxruntime onnxmltools")
        elif not os.path.exists("base_model_for_onnx.pkl"):
            st.warning("Save models first")
        else:
            if st.button("Convert to ONNX", type="primary"):
                with st.spinner("Converting..."):
                    with open("base_model_for_onnx.pkl", "rb") as f:
                        md = pickle.load(f)

                    model = md["model"]
                    nf = md["n_features_processed"]

                    initial_type = [("input", FloatTensorType([None, nf]))]
                    onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset=12,
                                                 options={type(model): {"zipmap": False}})

                    with open("mule_detection_model.onnx", "wb") as f:
                        f.write(onnx_model.SerializeToString())

                    loaded = onnx.load("mule_detection_model.onnx")
                    onnx.checker.check_model(loaded)

                st.success("ONNX conversion and validation successful")

                col1, col2, col3 = st.columns(3)
                with col1:
                    s = os.path.getsize("mule_detection_model.onnx") / (1024 * 1024)
                    render_kpi("bolt", "ONNX Size", f"{s:.1f} MB", "", "green")
                with col2:
                    render_kpi("input", "Features", f"{nf}", "", "orange")
                with col3:
                    render_kpi("category", "Classes", f"{len(md['class_names'])}", "", "teal")

            if os.path.exists("mule_detection_model.onnx"):
                with open("mule_detection_model.onnx", "rb") as f:
                    st.download_button("Download ONNX Model", data=f, file_name="mule_detection_model.onnx")

    with tab3:
        render_section("Test with Fresh Data", "science")
        render_section_note("Use a fresh synthetic sample to confirm the saved artifacts still score end to end after export.")

        if not os.path.exists("best_model.pkl"):
            st.warning("Save models first")
        else:
            test_seed = st.number_input("Test Seed", value=999, min_value=1, max_value=99999)

            if st.button("Generate and Test", type="primary"):
                from config import CONFIG
                from data_ingestion import DataIngestion
                from entity_resolution import EntityResolution
                from feature_engineering import FeatureEngineering
                import random

                original_seed = CONFIG["random_state"]
                CONFIG["random_state"] = test_seed
                np.random.seed(test_seed)
                random.seed(test_seed)

                progress = st.progress(0)

                with st.spinner("Generating data..."):
                    ing = DataIngestion()
                    ent = EntityResolution()
                    feat = FeatureEngineering()

                    raw = ing.generate_raw_tables()
                    progress.progress(15)
                    txn = ing.generate_transaction_tables(raw)
                    progress.progress(30)
                    ev = ent.build_entity_views(raw)
                    progress.progress(40)
                    events = ent.build_unified_events(txn)
                    progress.progress(50)
                    sv = ent.build_single_view(events, ev)
                    progress.progress(60)
                    clean = feat.run_eda_and_imputation(sv)
                    progress.progress(75)
                    test_data = feat.feature_engineering(clean)
                    progress.progress(90)

                CONFIG["random_state"] = original_seed

                with st.spinner("Running predictions..."):
                    with open("best_model.pkl", "rb") as f:
                        md = pickle.load(f)

                    X = test_data.reindex(columns=md["feature_cols"]).copy()
                    dt_cols = X.select_dtypes(include=["datetime64", "datetimetz", "datetime64[ns]", "timedelta64"]).columns.tolist()
                    if dt_cols:
                        X = X.drop(columns=dt_cols)

                    obj_cols = X.select_dtypes(include=["object"]).columns.tolist()
                    for c in obj_cols:
                        X[c] = X[c].apply(lambda x: str(x) if pd.notna(x) else "MISSING")

                    drop = [c for c in obj_cols if X[c].nunique() > 1000 or X[c].nunique() <= 1]
                    if drop:
                        X = X.drop(columns=drop)

                    try:
                        X_proc = md["preprocessor"].transform(X)
                    except Exception:
                        X_proc = md["preprocessor"].fit_transform(X)

                    t0 = time.time()
                    proba = md["calibrated_model"].predict_proba(X_proc)
                    inf_time = time.time() - t0

                    pred_enc = np.argmax(proba, axis=1)
                    pred_labels = md["label_encoder"].inverse_transform(pred_enc)
                    confidence = np.max(proba, axis=1)

                progress.progress(100)

                from sklearn.metrics import accuracy_score, f1_score, classification_report

                actual = test_data["label"].astype(str).values
                le = md["label_encoder"]

                try:
                    actual_enc = le.transform(actual)

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        render_kpi("data_array", "Records", f"{len(test_data):,}", "", "orange")
                    with col2:
                        render_kpi("check", "Accuracy", f"{accuracy_score(actual_enc, pred_enc):.4f}", "", "green")
                    with col3:
                        render_kpi("star", "Macro F1", f"{f1_score(actual_enc, pred_enc, average='macro'):.4f}", "", "gold")
                    with col4:
                        render_kpi("speed", "Inference", f"{inf_time:.3f}s", "", "teal")

                    report = classification_report(actual_enc, pred_enc, target_names=le.classes_, output_dict=True)
                    st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

                except Exception as e:
                    st.warning(f"Evaluation error: {e}")

                dist = pd.Series(pred_labels).value_counts().reset_index()
                dist.columns = ["Label", "Count"]
                fig = px.bar(dist, x="Label", y="Count", color="Label")
                fig = apply_dark_theme(fig)
                fig.update_layout(height=400, showlegend=False, title="Prediction Distribution")
                st.plotly_chart(fig, use_container_width=True)


def page_monitoring():
    render_top_bar("System Monitoring", "Pipeline execution status and performance metrics")
    render_page_intro("Track which stages are complete, how long each one took, and which files were saved so the workspace stays transparent.")

    steps = [
        ("Data Gen", st.session_state.raw_tables is not None),
        ("Entity Res", st.session_state.single_view is not None),
        ("Features", st.session_state.feature_df is not None),
        ("Graph", st.session_state.graph_feature_df is not None),
        ("Training", st.session_state.model6_artifacts is not None),
        ("Alerts", st.session_state.alert_output is not None),
        ("Feedback", st.session_state.feedback_outputs is not None),
        ("Export", os.path.exists("best_model.pkl")),
    ]

    render_section("Step Status", "check_circle")
    render_section_note("These cards show which stages are ready and which ones still need to be run.")

    cols = st.columns(8)
    for i, (name, done) in enumerate(steps):
        with cols[i]:
            color = "green" if done else "red"
            icon = "check_circle" if done else "pending"
            render_kpi(icon, name, "Done" if done else "Pending", "", color)

    if st.session_state.step_times:
        render_section("Execution Timeline", "timeline")
        render_section_note("The bar chart compares runtime across the completed stages so you can spot slow steps quickly.")

        data = pd.DataFrame([{"Step": k, "Time": round(v, 2)} for k, v in st.session_state.step_times.items()])
        fig = px.bar(data, y="Step", x="Time", orientation="h",
                     color="Time", color_continuous_scale=[[0, "#F7E6DA"], [0.5, "#D04A02"], [1, "#FFB600"]])
        fig = apply_dark_theme(fig)
        fig.update_layout(height=max(300, len(data) * 35), showlegend=False, title="Execution Times (seconds)",
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

        total = sum(st.session_state.step_times.values())
        col1, col2 = st.columns(2)
        with col1:
            render_kpi("timer", "Total Time", f"{total:.1f}s", f"{total/60:.1f} minutes", "orange")
        with col2:
            render_kpi("speed", "Avg Step", f"{total/len(st.session_state.step_times):.1f}s", "", "teal")

    render_section("Saved Files", "folder")
    render_section_note("Artifacts written to disk are listed below with an availability check and file size when present.")

    files = {
        "best_model.pkl": "Complete model",
        "base_model_for_onnx.pkl": "ONNX base",
        "mule_detection_model.onnx": "ONNX model",
        "feature_importance.csv": "Features",
    }

    for fname, desc in files.items():
        exists = os.path.exists(fname)
        size = f"{os.path.getsize(fname) / (1024*1024):.2f} MB" if exists else ""
        badge = "success" if exists else "danger"
        badge_text = "Available" if exists else "Missing"
        st.markdown(f"""
        <div class="table-row">
            <span class="name">{fname} <span style="color: #666; font-weight: 400;">({desc})</span></span>
            <span><span class="badge {badge}">{badge_text}</span> <span class="info">{size}</span></span>
        </div>
        """, unsafe_allow_html=True)


def main():
    init_session_state()
    inject_theme_css()

    with st.sidebar:
        render_sidebar_v2()

    page = st.session_state.page

    if page == "Overview":
        page_overview()
    elif page == "1. Data Generation":
        page_data_generation()
    elif page == "2. Entity Resolution":
        page_entity_resolution()
    elif page == "3. Feature Engineering":
        page_feature_engineering()
    elif page == "4. Graph Analytics":
        page_graph_analytics()
    elif page == "5. Model Training":
        page_model_training()
    elif page == "Model Inference":
        page_model_inference()
    elif page == "6. Alert Engine":
        page_alerts()
    elif page == "7. Feedback Loop":
        page_feedback()
    elif page == "8. Export":
        page_export()
    elif page == "Monitoring":
        page_monitoring()


if __name__ == "__main__":
    main()
