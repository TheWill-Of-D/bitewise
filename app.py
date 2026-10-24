from __future__ import annotations

import os
import hashlib
import logging
import sqlite3
import time
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Iterator
from html import escape

import pandas as pd
import streamlit as st

from recognition import predict_image, model_status, model_cache_key, open_image


ROOT = Path(__file__).parent
DB_PATH = Path(os.environ.get("BITEWISE_DB_PATH", ROOT / "bitewise.sqlite3"))
NUTRITION_PATH = ROOT / "data" / "nutrition.csv"

st.set_page_config(page_title="BiteWise · Meal companion", page_icon="🥢", layout="wide")

STYLE = """
<style>
:root { --ink:#18362f; --muted:#687b74; --green:#347a60; --lime:#c9e7a5; --paper:#f6f8f2; --line:#e6ebe3; }
html, body, [class*="css"] { font-family:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
.stApp { background:var(--paper); color:var(--ink); }
.block-container { max-width:1220px; padding-top:1.7rem; padding-bottom:3rem; }
h1,h2,h3 { font-family:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif !important; color:var(--ink) !important; letter-spacing:-.035em; }
h1 { font-size:2.45rem !important; }
p, label, .stMarkdown { color:var(--ink); }
div[data-testid="stMetric"] { background:white; border:1px solid var(--line); border-radius:18px; padding:18px 20px; box-shadow:0 3px 14px rgba(29,64,51,.035); }
div[data-testid="stMetricLabel"] { color:var(--muted); font-size:.9rem; }
div[data-testid="stMetricValue"] { color:var(--ink); font-family:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
div[data-testid="stForm"] { background:white; border:1px solid var(--line); border-radius:20px; padding:1.2rem 1.4rem; }
div[data-testid="stFileUploader"] section { background:#fbfcf9; border:1px dashed #b7c9bb; border-radius:14px; }
.hero { background:linear-gradient(120deg,#183c33 0%,#285c49 64%,#39765a 100%); border-radius:24px; padding:27px 32px; margin:0 0 20px; color:#f7fbf3; position:relative; overflow:hidden; }
.hero:after { content:''; position:absolute; right:35px; top:25px; width:110px; height:110px; border:2px solid #c9e7a5; border-radius:50%; box-shadow:0 0 0 14px rgba(201,231,165,.15); opacity:.14; pointer-events:none; }
.hero .eyebrow { color:#c9e7a5; font-size:.76rem; font-weight:700; letter-spacing:.16em; text-transform:uppercase; }
.hero h1 { color:#fff !important; margin:.25rem 0 .25rem; }
.hero p { color:#d9e8dc; margin:0; max-width:680px; font-size:1.02rem; }
.section-kicker { color:#71827a; font-weight:700; text-transform:uppercase; letter-spacing:.12em; font-size:.72rem; margin:0 0 .45rem; }
.panel-title { font:700 1.25rem ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:var(--ink); margin:0 0 .25rem; }
.panel-copy { color:var(--muted); font-size:.92rem; margin:0 0 1rem; }
.meal-card { background:white; border:1px solid var(--line); border-radius:15px; padding:13px 16px; margin:9px 0; }
.meal-name { font-weight:700; color:var(--ink); font-size:1rem; }
.meal-sub { color:var(--muted); font-size:.82rem; margin-top:3px; }
.protein-pill { background:#edf5e6; color:#3c704b; border-radius:20px; padding:5px 10px; font-size:.8rem; font-weight:700; white-space:nowrap; }
.tiny-note { color:var(--muted); font-size:.78rem; }
.stButton>button[kind="primary"] { background:#347a60; border:0; border-radius:11px; min-height:2.8rem; font-weight:700; }
.stButton>button[kind="primary"] p { color:white; }
.stButton>button { border-radius:10px; }
div[data-testid="stSidebar"] { background:#edf2e9; }
div[data-testid="stSidebar"] h2 { font-size:1.2rem !important; }
footer {visibility:hidden;}
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def init_db() -> None:
    with connect() as con:
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY, value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS meals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                eaten_on TEXT NOT NULL,
                eaten_at TEXT NOT NULL,
                food TEXT NOT NULL,
                grams REAL NOT NULL,
                calories REAL NOT NULL,
                protein REAL NOT NULL,
                carbs REAL NOT NULL,
                fat REAL NOT NULL,
                source TEXT NOT NULL
            );
            """
        )


def setting(key: str, default: str) -> str:
    with connect() as con:
        row = con.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default


def save_setting(key: str, value: str) -> None:
    with connect() as con:
        con.execute("INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))


def nutrition_data() -> pd.DataFrame:
    return pd.read_csv(NUTRITION_PATH).sort_values("food")


def today_meals(day: date) -> pd.DataFrame:
    with connect() as con:
        return pd.read_sql_query("SELECT * FROM meals WHERE eaten_on=? ORDER BY id DESC", con, params=(day.isoformat(),))


def estimate(row: pd.Series, grams: float) -> dict[str, float]:
    factor = grams / 100.0
    return {key: float(row[key]) * factor for key in ("calories", "protein", "carbs", "fat")}


def add_meal(day: date, food: str, grams: float, values: dict[str, float], source: str) -> None:
    with connect() as con:
        con.execute(
            "INSERT INTO meals(eaten_on,eaten_at,food,grams,calories,protein,carbs,fat,source) VALUES(?,?,?,?,?,?,?,?,?)",
            (day.isoformat(), datetime.now().strftime("%H:%M"), food, grams, values["calories"], values["protein"], values["carbs"], values["fat"], source),
        )


def remove_meal(meal_id: int) -> None:
    with connect() as con:
        con.execute("DELETE FROM meals WHERE id=?", (meal_id,))


init_db()
foods = nutrition_data()
food_names = foods["food"].tolist()
today = date.today()

with st.sidebar:
    st.markdown("## Your daily goal")
    age = st.number_input("Age", min_value=50, max_value=110, value=int(setting("age", "58")), step=1)
    weight = st.number_input("Weight (kg)", min_value=30.0, max_value=250.0, value=float(setting("weight", "65")), step=0.5)
    protein_goal = round(weight * 1.2)
    st.markdown(f"### {protein_goal} g protein / day")
    st.caption("Personalized estimate at 1.2 g per kg body weight. For general awareness, not medical advice.")
    if st.button("Save profile", width="stretch"):
        save_setting("age", str(age))
        save_setting("weight", str(weight))
        st.toast("Profile saved on this device")
    st.divider()
    ready, recognition_label = model_status()
    st.markdown("### Photo recognition")
    st.caption(recognition_label)
    if ready and "Food-101" in recognition_label:
        st.caption("General food categories. Local hawker dishes may need manual selection. This is a pretrained baseline, not a model trained by BiteWise.")
    st.divider()
    st.caption("Stored locally. Photos and meal logs stay on this device. No account required.")

st.markdown(
    "<div class='hero'><div class='eyebrow'>Your everyday nutrition companion</div><h1>BiteWise</h1><p>A little help keeping an eye on protein. Add a meal, check the estimate, and make it your own.</p></div>",
    unsafe_allow_html=True,
)

logged = today_meals(today)
protein_total = float(logged["protein"].sum()) if not logged.empty else 0.0
calories_total = float(logged["calories"].sum()) if not logged.empty else 0.0
carbs_total = float(logged["carbs"].sum()) if not logged.empty else 0.0
fat_total = float(logged["fat"].sum()) if not logged.empty else 0.0
meal_count = len(logged)

metric_cols = st.columns(4)
metric_cols[0].metric("Protein today", f"{protein_total:.0f} g", help=f"Daily target: {protein_goal} g")
metric_cols[1].metric("Calories", f"{calories_total:.0f} kcal")
metric_cols[2].metric("Meals logged", str(meal_count))
metric_cols[3].metric("To your protein goal", f"{max(protein_goal-protein_total, 0):.0f} g" if protein_total < protein_goal else "Goal reached")

progress = min(protein_total / protein_goal, 1.0) if protein_goal else 0
st.progress(progress, text=f"{protein_total:.0f} g protein · {round(progress*100)}% of today's goal")
st.write("")

left, right = st.columns([1.08, .92], gap="large")
with left:
    st.markdown("<div class='section-kicker'>Add a meal</div><div class='panel-title'>What did you enjoy?</div><div class='panel-copy'>Add a photo, confirm the dish, then adjust the serving size.</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Meal photo (optional)", type=["jpg", "jpeg", "png", "webp"], help="Your photo is processed by this local app and is not sent to an online service.")
    if uploaded:
        image_bytes = uploaded.getvalue()
        try:
            current_model_key = model_cache_key()
        except ValueError:
            current_model_key = "unavailable"
        image_digest = hashlib.sha256(image_bytes).hexdigest() + current_model_key
        if st.session_state.get("model_image_digest") != image_digest:
            st.session_state["model_candidates"] = []
            st.session_state["recognition_error"] = None
            st.session_state["confirmed_food"] = None
            try:
                open_image(image_bytes)
                is_ready, _ = model_status()
                if is_ready:
                    with st.spinner("Looking at your meal…"):
                        started = time.perf_counter()
                        st.session_state["model_candidates"] = predict_image(image_bytes)
                        st.session_state["recognition_seconds"] = time.perf_counter() - started
                    candidates = st.session_state["model_candidates"]
                    if candidates and candidates[0][0] in food_names and candidates[0][1] >= .6:
                        st.session_state["confirmed_food"] = candidates[0][0]
            except Exception as exc:
                logging.getLogger(__name__).warning("Meal photo could not be recognized: %s", exc)
                st.session_state["recognition_error"] = "Recognition could not process this photo. Choose a dish below, or try another photo."
            st.session_state["model_image_digest"] = image_digest
        try:
            st.image(open_image(image_bytes), caption="Meal photo", width="stretch")
        except Exception:
            st.warning("This file could not be opened as a photo. Please upload a valid JPG, PNG, or WebP image.")
        candidates = st.session_state.get("model_candidates", [])
        if st.session_state.get("recognition_error"):
            st.warning(st.session_state["recognition_error"])
        elif candidates:
            name, score = candidates[0]
            if score < .6:
                st.warning(f"Uncertain suggestion: {name} · {score:.0%} model score. Please choose the correct dish below.")
            else:
                st.info(f"Suggested dish: {name} · {score:.0%} model score. Please confirm or correct it below.")
            if name not in food_names:
                st.warning(f"There is no nutrition entry linked to {name}. Choose the actual dish below; a similar-looking dish can have different nutrients.")
            st.caption("Other possibilities: " + " · ".join(f"{name} ({score:.0%})" for name, score in candidates[1:]))
            st.caption("Model scores describe relative confidence, not a guarantee that the dish is correct.")
            if "recognition_seconds" in st.session_state:
                st.caption(f"Processed on this device in {st.session_state['recognition_seconds']:.2f} seconds.")
        else:
            st.info("Photo added. Recognition is not configured yet. Choose the dish below to continue.")
    else:
        st.caption("A photo is optional. It stays on this device and is not sent to an online service.")

    selected_food = st.selectbox("Confirm or choose the dish", food_names, index=None, placeholder="Search for your dish…", key="confirmed_food")
    portion = st.number_input("Serving size (g)", min_value=25, max_value=1500, value=250, step=25, help="A rough estimate is fine. Nutrition scales with serving size.")
    if selected_food:
        nutrition_row = foods.loc[foods["food"] == selected_food].iloc[0]
        values = estimate(nutrition_row, portion)
        st.markdown(
            f"<div class='meal-card'><div class='meal-name'>Estimated for {portion} g</div><div class='meal-sub'>{values['calories']:.0f} kcal &nbsp; · &nbsp; <b>{values['protein']:.1f} g protein</b> &nbsp; · &nbsp; {values['carbs']:.1f} g carbs &nbsp; · &nbsp; {values['fat']:.1f} g fat</div></div>",
            unsafe_allow_html=True,
        )
        source_note = "Illustrative prototype values, not verified reference data." if "starter" in str(nutrition_row['source']).lower() else "Nutrition and portions are estimates."
        st.caption(f"Source: {nutrition_row['source']}. {source_note}")
    if st.button("Add to today's meals", type="primary", width="stretch", disabled=selected_food is None):
        add_meal(today, selected_food, float(portion), values, str(nutrition_row["source"]))
        st.toast(f"{selected_food} added to today's meals")
        st.rerun()

with right:
    st.markdown("<div class='section-kicker'>Today's plate</div><div class='panel-title'>Your meal log</div><div class='panel-copy'>A simple view of what you've recorded today.</div>", unsafe_allow_html=True)
    if logged.empty:
        st.markdown("<div class='meal-card'><div class='meal-name'>No meals yet</div><div class='meal-sub'>Your meals will appear here as you add them.</div></div>", unsafe_allow_html=True)
    else:
        for _, meal in logged.iterrows():
            c1, c2 = st.columns([.75,.25], gap="small", vertical_alignment="center")
            with c1:
                st.markdown(f"<div class='meal-card'><div style='display:flex;justify-content:space-between;gap:12px'><div><div class='meal-name'>{escape(meal['food'])}</div><div class='meal-sub'>{meal['eaten_at']} · {meal['grams']:.0f} g · {meal['calories']:.0f} kcal</div></div><span class='protein-pill'>{meal['protein']:.1f} g protein</span></div><div class='tiny-note'>Source: {escape(meal['source'])}</div></div>", unsafe_allow_html=True)
            with c2:
                if st.button("Remove", key=f"remove_{meal['id']}", help="Remove this meal"):
                    remove_meal(int(meal["id"]))
                    st.rerun()
    st.markdown("<div class='section-kicker' style='margin-top:22px'>Nutrition snapshot</div>", unsafe_allow_html=True)
    if meal_count == 0:
        st.info("Add your first meal to start tracking today's protein.")
    elif protein_total < protein_goal:
        st.info(f"You have {protein_goal-protein_total:.1f} g left to your protein target, based on the meals logged so far. Look for a source of protein in your next meal, such as fish, eggs, tofu, or beans.")
    else:
        st.success("Your logged meals have reached today's estimated protein target.")
    st.markdown(f"<div class='meal-card'><div class='meal-name'>Protein first, progress over perfection</div><div class='meal-sub'>You've logged {protein_total:.1f} g protein, {calories_total:.0f} kcal, {carbs_total:.1f} g carbs, and {fat_total:.1f} g fat today. These values are estimates to support everyday awareness.</div></div>", unsafe_allow_html=True)

st.divider()
st.markdown("<div style='text-align:center;color:#71827a;font-size:.82rem'>BiteWise · Local prototype · Your data is stored on this device</div>", unsafe_allow_html=True)
