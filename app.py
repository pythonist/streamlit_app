import streamlit as st
import streamlit.components.v1 as components
import pickle
import numpy as np
import pandas as pd
import time
import json
import os
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

from config import CONFIG

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
        for item in nav_items[1:9]:
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

    render_section("Configuration", "settings")

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

        all_tables = {**raw_tables, **txn_tables}
        selected = st.selectbox("Select table", list(all_tables.keys()), label_visibility="collapsed")

        if selected:
            df = all_tables[selected]

            tab1, tab2, tab3 = st.tabs(["Preview", "Schema", "Statistics"])

            with tab1:
                st.dataframe(df.head(200), use_container_width=True, height=400)

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
                st.dataframe(df.describe().transpose().round(4), use_container_width=True, height=400)


def page_entity_resolution():
    render_top_bar("Entity Resolution", "Build entity views, unified events, and customer single view")
    render_progress_steps(2)

    if st.session_state.raw_tables is None:
        st.warning("Complete Step 1 first: Generate Data")
        return

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

        cols = st.columns(len(st.session_state.entity_views))
        for i, (name, df) in enumerate(st.session_state.entity_views.items()):
            with cols[i]:
                short_name = name.replace("_view", "").title()
                render_kpi("group", short_name, f"{df.shape[0]:,}", f"{df.shape[1]} columns", "orange")

    if st.session_state.events is not None:
        render_section("Unified Events", "timeline")

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

    if st.session_state.single_view is not None:
        render_section("Single View Preview", "table_chart")
        st.dataframe(st.session_state.single_view.head(100), use_container_width=True, height=300)


def page_feature_engineering():
    render_top_bar("Feature Engineering", "EDA, imputation, and advanced feature generation")
    render_progress_steps(3)

    if st.session_state.single_view is None:
        st.warning("Complete Step 2 first: Entity Resolution")
        return

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

    if st.session_state.feature_df is not None:
        df = st.session_state.feature_df

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            render_kpi("functions", "Total Features", f"{df.shape[1]}", "", "orange")
        with col2:
            render_kpi("pin", "Numeric", f"{len(df.select_dtypes(include=[np.number]).columns)}", "", "teal")
        with col3:
            render_kpi("text_fields", "Categorical", f"{len(df.select_dtypes(include=['object']).columns)}", "", "gold")
        with col4:
            render_kpi("data_array", "Records", f"{df.shape[0]:,}", "", "green")

        tab1, tab2, tab3 = st.tabs(["Data Preview", "Feature Distributions", "Label Analysis"])

        with tab1:
            st.dataframe(df.head(200), use_container_width=True, height=400)

        with tab2:
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            sel = st.selectbox("Select feature", num_cols[:50])
            if sel:
                fig = px.histogram(df, x=sel, nbins=50, color_discrete_sequence=[PWC_COLORS["primary"]])
                fig = apply_dark_theme(fig)
                fig.update_layout(height=400, title=f"Distribution: {sel}")
                st.plotly_chart(fig, use_container_width=True)

        with tab3:
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
                        fig = px.box(df, x="label", y="amount", color="label")
                        fig = apply_dark_theme(fig)
                        fig.update_layout(height=400, showlegend=False, title="Amount by Label")
                        st.plotly_chart(fig, use_container_width=True)


def page_graph_analytics():
    render_top_bar("Graph Analytics", "Network analysis, community detection, and mule ring identification")
    render_progress_steps(4)

    if st.session_state.feature_df is None:
        st.warning("Complete Step 3 first: Feature Engineering")
        return

    render_section("Configuration", "tune")

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
        render_network_graph(st.session_state.graph_feature_df, st.session_state.ring_df)

    if st.session_state.ring_df is not None and isinstance(st.session_state.ring_df, pd.DataFrame) and len(st.session_state.ring_df) > 0:
        render_section("Detected Rings", "toll")

        ring_df = st.session_state.ring_df

        col1, col2 = st.columns(2)
        with col1:
            if "ring_member_count" in ring_df.columns:
                fig = px.histogram(ring_df, x="ring_member_count", nbins=20,
                                   color_discrete_sequence=[PWC_COLORS["accent_rose"]])
                fig = apply_dark_theme(fig)
                fig.update_layout(height=350, title="Ring Size Distribution")
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            if "ring_risk_score" in ring_df.columns:
                fig = px.histogram(ring_df, x="ring_risk_score", nbins=20,
                                   color_discrete_sequence=[PWC_COLORS["primary"]])
                fig = apply_dark_theme(fig)
                fig.update_layout(height=350, title="Ring Risk Score Distribution")
                st.plotly_chart(fig, use_container_width=True)

        st.dataframe(ring_df.head(100), use_container_width=True, height=300)


def page_model_training():
    render_top_bar("Model Training Studio", "Configure, train, and evaluate classification models")
    render_progress_steps(5)

    working_df = st.session_state.graph_feature_df if st.session_state.graph_feature_df is not None else st.session_state.feature_df

    if working_df is None:
        st.warning("Complete previous steps first")
        return

    render_section("Algorithm Selection", "model_training")

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
            report = classification_report(y_test, test_pred, target_names=le.classes_, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True, height=400)

        with tab2:
            cm = confusion_matrix(y_test, test_pred)
            cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
            fig = px.imshow(cm_df, text_auto=True, color_continuous_scale=[[0, "#F8ECE1"], [0.5, "#D04A02"], [1, "#FFB600"]])
            fig = apply_dark_theme(fig)
            fig.update_layout(height=600, title="Confusion Matrix")
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
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


def page_alerts():
    render_top_bar("Alert Engine", "Decision engine and intelligent alert packaging")
    render_progress_steps(6)

    if st.session_state.model6_artifacts is None:
        st.warning("Train models first in Step 5")
        return

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
        st.dataframe(ao.head(500), use_container_width=True, height=400)

        csv = ao.to_csv(index=False)
        st.download_button("Download Alerts", data=csv, file_name="alerts.csv", mime="text/csv")


def page_feedback():
    render_top_bar("Feedback Loop", "Weak supervision and model performance feedback")
    render_progress_steps(7)

    if st.session_state.alert_output is None:
        st.warning("Generate alerts first in Step 6")
        return

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
        outputs = st.session_state.feedback_outputs
        if isinstance(outputs, dict):
            for key, val in outputs.items():
                if isinstance(val, pd.DataFrame):
                    st.markdown(f'<div class="section-title">{key}</div>', unsafe_allow_html=True)
                    st.dataframe(val.head(100), use_container_width=True, height=250)


def page_export():
    render_top_bar("Save and Export", "Save models, convert to ONNX, and validate with fresh data")
    render_progress_steps(8)

    tab1, tab2, tab3 = st.tabs(["Save Models", "ONNX Conversion", "Fresh Data Test"])

    with tab1:
        render_section("Save Model Artifacts", "save")

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

    cols = st.columns(8)
    for i, (name, done) in enumerate(steps):
        with cols[i]:
            color = "green" if done else "red"
            icon = "check_circle" if done else "pending"
            render_kpi(icon, name, "Done" if done else "Pending", "", color)

    if st.session_state.step_times:
        render_section("Execution Timeline", "timeline")

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
