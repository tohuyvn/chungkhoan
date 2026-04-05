import base64
from pathlib import Path

# chay thu : streamlit run main.py
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression

_BASE = Path(__file__).resolve().parent
stocks = ["VCB", "VIC", "VHM", "BID", "HPG"]


def _load_df() -> pd.DataFrame:
    xlsx = _BASE / "stock_cleaned.xlsx"
    csv = _BASE / "stock_cleaned.csv"
    if xlsx.exists():
        return pd.read_excel(xlsx)
    if csv.exists():
        return pd.read_csv(csv)
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=800, freq="B")
    bases = {"VCB": 85.0, "VIC": 42.0, "VHM": 38.0, "BID": 48.0, "HPG": 28.0}
    out = {"Date": dates}
    for sym, base in bases.items():
        walk = np.cumsum(rng.normal(0, 0.8, len(dates)))
        out[sym] = np.clip(base + walk, 5, None)
    return pd.DataFrame(out)


def _background_image_css_url() -> str | None:
    """Trả về url('data:...') nếu tìm thấy ảnh nền trong thư mục dự án."""
    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }
    names = ("background", "bg", "hero")
    folders = [_BASE / "assets", _BASE]
    max_bytes = 3 * 1024 * 1024
    for folder in folders:
        if folder.name == "assets" and not folder.is_dir():
            continue
        for stem in names:
            for ext, m in mime.items():
                path = folder / f"{stem}{ext}"
                if not path.is_file():
                    continue
                raw = path.read_bytes()
                if len(raw) > max_bytes:
                    continue
                b64 = base64.b64encode(raw).decode("ascii")
                return f"url('data:{m};base64,{b64}')"
    return None


def _inject_custom_css() -> None:
    bg = _background_image_css_url()
    if bg:
        app_bg = f"""
        [data-testid="stAppViewContainer"] {{
            background-image:
                linear-gradient(
                    165deg,
                    rgba(15, 23, 42, 0.72) 0%,
                    rgba(30, 41, 59, 0.55) 45%,
                    rgba(15, 23, 42, 0.68) 100%
                ),
                {bg};
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        """
    else:
        app_bg = """
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(
                165deg,
                #fffefb 0%,
                #f0f9ff 35%,
                #e0f2fe 70%,
                #f8fafc 100%
            );
        }
        """

    st.markdown(
        f"""
        <style>
        @import url("https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,600;0,9..40,700&display=swap");

        :root {{
            --accent: #059669;
            --accent-soft: rgba(5, 150, 105, 0.1);
            --text-main: #0f172a;
            --text-muted: #475569;
            --border: #e2e8f0;
            --card: rgba(255, 255, 255, 0.97);
        }}

        html, body, [data-testid="stAppViewContainer"] {{
            font-family: "DM Sans", system-ui, sans-serif;
            color: var(--text-main);
        }}

        {app_bg}

        [data-testid="stAppViewContainer"] .main {{
            background: transparent;
        }}

        .main .block-container {{
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1200px;
            background: var(--card);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 1);
            box-shadow: 0 12px 48px rgba(15, 23, 42, 0.12);
        }}

        /* Toàn bộ vùng app: chữ tối rõ trên nền sáng */
        .main .block-container,
        .main .block-container * {{
            color-scheme: light;
        }}

        h1 {{
            font-weight: 700 !important;
            letter-spacing: -0.03em;
            font-size: 2.15rem !important;
            color: #0f172a !important;
            border-bottom: 2px solid var(--border);
            padding-bottom: 0.75rem;
            margin-bottom: 1.25rem !important;
        }}

        /* Tiêu đề phụ / markdown: nền sáng để không chìm vào ảnh nền */
        h2, h3 {{
            font-weight: 600 !important;
            color: #0f172a !important;
            background: rgba(255, 255, 255, 0.98);
            padding: 0.5rem 0.85rem;
            border-radius: 10px;
            border: 1px solid var(--border);
            box-shadow: 0 2px 10px rgba(15, 23, 42, 0.06);
            width: fit-content;
            max-width: 100%;
        }}

        /* st.title / st.subheader / st.header (Streamlit) */
        [data-testid="stHeading"] {{
            background: rgba(255, 255, 255, 0.98) !important;
            padding: 0.55rem 0.95rem !important;
            border-radius: 10px !important;
            border: 1px solid var(--border) !important;
            box-shadow: 0 2px 10px rgba(15, 23, 42, 0.06) !important;
            width: fit-content !important;
            max-width: 100% !important;
        }}

        [data-testid="stHeading"] h1,
        [data-testid="stHeading"] h2,
        [data-testid="stHeading"] h3 {{
            color: #0f172a !important;
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            padding: 0 !important;
            margin: 0 !important;
            width: auto !important;
        }}

        [data-testid="stHeading"] * {{
            color: #0f172a !important;
        }}

        .main [data-testid="stMarkdownContainer"] p,
        .main [data-testid="stMarkdownContainer"] li,
        .main [data-testid="stMarkdownContainer"] span {{
            color: #0f172a !important;
        }}

        .main [data-testid="stMarkdownContainer"] h2,
        .main [data-testid="stMarkdownContainer"] h3 {{
            color: #0f172a !important;
        }}

        p, span, label {{
            color: inherit;
        }}

        [data-testid="stCaption"] {{
            color: #475569 !important;
        }}

        [data-testid="stHeader"] {{
            background: rgba(255, 255, 255, 0.92);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid var(--border);
        }}

        div[data-testid="stDecoration"] {{
            background: linear-gradient(90deg, #059669, #0ea5e9);
        }}

        /* Nhãn widget: nền sáng cố định (label thường nằm trên vùng hở ảnh nền tối) */
        [data-testid="stWidgetLabel"] {{
            background: rgba(255, 255, 255, 0.98) !important;
            padding: 0.45rem 0.8rem !important;
            border-radius: 10px !important;
            border: 1px solid #e2e8f0 !important;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.18) !important;
            display: block !important;
            width: fit-content !important;
            max-width: 100% !important;
            margin-bottom: 0.35rem !important;
        }}

        [data-testid="stWidgetLabel"] p,
        [data-testid="stWidgetLabel"] label {{
            font-weight: 600 !important;
            color: #0f172a !important;
            margin: 0 !important;
        }}

        .stSelectbox label,
        .stDateInput label,
        label[data-testid="stWidgetLabel"] {{
            font-weight: 600 !important;
            color: #0f172a !important;
        }}

        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] input {{
            border-radius: 12px !important;
            border-color: var(--border) !important;
            color: #0f172a !important;
            background-color: #ffffff !important;
        }}

        div[data-baseweb="select"] span {{
            color: #0f172a !important;
        }}

        /* Banner dự đoán (st.markdown): nền trắng, chữ vàng + số đỏ lớn gấp 3 */
        .main [data-testid="stMarkdownContainer"] .pred-banner {{
            background: #ffffff !important;
            border: 2px solid #facc15 !important;
            border-radius: 14px !important;
            padding: 0.85rem 1.1rem !important;
            box-shadow: 0 2px 14px rgba(0, 0, 0, 0.12) !important;
            display: flex !important;
            align-items: baseline !important;
            flex-wrap: wrap !important;
            gap: 0.35rem 0.6rem !important;
            margin: 0.5rem 0 1rem 0 !important;
        }}

        .main [data-testid="stMarkdownContainer"] .pred-banner__label {{
            color: #ca8a04 !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
            line-height: 1.3 !important;
        }}

        .main [data-testid="stMarkdownContainer"] .pred-banner__value {{
            color: #dc2626 !important;
            font-weight: 700 !important;
            font-size: 3rem !important;
            line-height: 1 !important;
            background: #ffffff !important;
        }}

        [data-testid="stAlert"] {{
            border-radius: 14px !important;
        }}

        /* st.info: nền và chữ rõ (success đã tách style phía trên) */
        [data-testid="stAlert"] [data-testid="stMarkdownContainer"] p {{
            color: #0f172a !important;
        }}

        [data-testid="stNotificationContent"] p {{
            color: #0f172a !important;
        }}

        [data-testid="stDataFrame"] {{
            border: 1px solid var(--border);
            border-radius: 14px;
            overflow: hidden;
            box-shadow: 0 2px 12px rgba(15, 23, 42, 0.06);
            background: #ffffff !important;
        }}

        [data-testid="stDataFrame"] * {{
            color: #0f172a !important;
        }}

        /* Plotly trong khung sáng */
        .js-plotly-plot .plotly .main-svg text {{
            fill: #334155 !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="Dự đoán chứng khoán", layout="wide")
_inject_custom_css()

# ===== LOAD DATA =====
df = _load_df()
df["Date"] = pd.to_datetime(df["Date"])
df = df[df["Date"] >= "2020-01-01"]

if not (_BASE / "stock_cleaned.xlsx").exists() and not (_BASE / "stock_cleaned.csv").exists():
    st.info(
        "Chưa có `stock_cleaned.xlsx` hoặc `stock_cleaned.csv` — đang dùng dữ liệu mẫu để chạy thử."
    )

st.title("📊 Web dự đoán chứng khoán")

col1, col2 = st.columns(2)

with col1:
    stock = st.selectbox("Chọn cổ phiếu", stocks)

with col2:
    future_date = st.date_input("Chọn ngày dự đoán")

# ===== XỬ LÝ DATA =====
data = df[["Date", stock]].dropna().copy()
data["Days"] = (data["Date"] - data["Date"].min()).dt.days

if data.empty:
    st.error(f"Không có dữ liệu cho mã {stock}.")
    st.stop()

X = data[["Days"]]
y = data[stock]

# ===== TRAIN MODEL =====
model = LinearRegression()
model.fit(X, y)

# ===== DỰ ĐOÁN 1 NGÀY =====
future_days = (pd.to_datetime(future_date) - data["Date"].min()).days
pred = model.predict(pd.DataFrame([[future_days]], columns=["Days"]))[0]

# ===== DỰ ĐOÁN 7 NGÀY =====
last_day = X["Days"].max()
future_days_7 = np.arange(last_day + 1, last_day + 8)

future_dates_7 = [
    data["Date"].min() + pd.Timedelta(days=int(d)) for d in future_days_7
]

future_preds_7 = model.predict(pd.DataFrame(future_days_7, columns=["Days"]))

# ===== VẼ BIỂU ĐỒ =====
fig = go.Figure()

# dữ liệu thật
fig.add_trace(
    go.Scatter(
        x=data["Date"],
        y=y,
        mode="lines",
        name="Giá thật",
    )
)

# hồi quy
fig.add_trace(
    go.Scatter(
        x=data["Date"],
        y=model.predict(X),
        mode="lines",
        name="Hồi quy",
        line=dict(dash="dash"),
    )
)

# 7 ngày tiếp
fig.add_trace(
    go.Scatter(
        x=future_dates_7,
        y=future_preds_7,
        mode="lines+markers",
        name="Dự đoán 7 ngày",
        line=dict(color="#ca8a04"),
    )
)

# điểm dự đoán
fig.add_trace(
    go.Scatter(
        x=[future_date],
        y=[pred],
        mode="markers+text",
        name="Dự đoán",
        text=[f"{round(pred, 2)}"],
        textposition="top center",
        marker=dict(size=10, color="red"),
    )
)

fig.update_layout(
    title=f"Biểu đồ cổ phiếu {stock}",
    xaxis_title="Ngày",
    yaxis_title="Giá",
    template="plotly_white",
    height=600,
    paper_bgcolor="rgba(255,255,255,0.65)",
    plot_bgcolor="rgba(248,250,252,0.9)",
)

# ===== HIỂN THỊ =====
st.plotly_chart(fig, use_container_width=True)

_pred_txt = f"{round(pred, 2):.2f}"
st.markdown(
    f'<div class="pred-banner">'
    f'<span class="pred-banner__label">📈 Giá dự đoán ngày {future_date}:</span>'
    f'<span class="pred-banner__value">{_pred_txt}</span>'
    f"</div>",
    unsafe_allow_html=True,
)

# ===== BẢNG 7 NGÀY =====
st.subheader("📊 Dự đoán 7 ngày tiếp theo")

result_df = pd.DataFrame(
    {
        "Ngày": future_dates_7,
        "Giá dự đoán": np.round(future_preds_7, 2),
    }
)

st.dataframe(result_df)
