import os
import pandas as pd
from openai import OpenAI
import streamlit as st
import base64

BASE_DIR = os.path.dirname(__file__)

# --- Background image (now relative to the app folder) ---
background_path = os.path.join(BASE_DIR, "Background.jpg")
# --- Background Setup ---
def set_background(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url(data:image/jpeg;base64,{encoded});
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

# Header
st.markdown(
    """
    <div style="
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background-color: rgba(255, 215, 0, 0.75);
        padding: 15px;
        border-radius: 5px;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        color: #fff;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        margin-bottom: 20px;">
        Happy to recommend...
    </div>
    """,
    unsafe_allow_html=True
)
# background_path = r"C:/Users/happy/Documents/ironhack/RoboReview/ai-reviewer/Background.jpg"
set_background(background_path)
# --- Compute the folder where this app.py lives ---


# DATA_CSV = r"C:/Users/happy/Documents/ironhack/RoboReview/ai-reviewer/data/enriched_with_clusters_deployment.csv"
# --- Load deployment data ---
# DATA_CSV = r"C:/Users/happy/Documents/ironhack/RoboReview/ai-reviewer/happytorecommendcars/ai-reviewer/data/enriched_with_clusters_deployment.csv"

# df = pd.read_csv(DATA_CSV)


# --- Build a path to data/enriched_with_clusters_deployment.csv under that same folder ---
DATA_CSV = os.path.join(BASE_DIR, "data", "enriched_with_clusters_deployment.csv")

# --- Guard for missing file with a Streamlit error instead of a Python traceback ---
if not os.path.isfile(DATA_CSV):
    st.error(f"Data file not found at {DATA_CSV!r}")
    st.stop()

# --- Now load it ---
df = pd.read_csv(DATA_CSV)


# Verify required columns
required = ['make','model','assigned_topic','make_cluster_perc','assigned_topic_cluster_perc',
            'sentiment','sentiment_score','cluster','vehicle_title','review','text_for_clustering','strengths','weaknesses']
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Missing columns: {', '.join(missing)}")
    st.stop()

# --- OpenAI client ---
API_KEY = os.getenv('OPENAI_API_KEY')
if not API_KEY:
    st.error('Set OPENAI_API_KEY environment variable')
    st.stop()
client = OpenAI(api_key=API_KEY)

# --- UI Controls ---
# Safely generate make options as strings
top_make = sorted(df['make'].dropna().astype(str).unique().tolist())
# Assigned topics as strings, exclude 'other'
topics = sorted([t for t in df['assigned_topic'].dropna().astype(str).unique().tolist() if t.lower()!='other'])

# Styled headings for filters
st.markdown(
    """
    <style>
    .filter-label {
        font-weight: bold;
        color: white;
        font-size: 16px;
    }
    </style>
    """, unsafe_allow_html=True
)

# Multi-select for characteristics
st.markdown('<div class="filter-label">Choose your "car"acteristics:</div>', unsafe_allow_html=True)
selected_topic = st.multiselect('', topics)
# Multi-select for makes
st.markdown('<div class="filter-label">Narrow your car choices?</div>', unsafe_allow_html=True)
selected_makes = st.multiselect('', top_make)

# --- Filter data ---
filtered = df.copy()
# Cast to str for safe comparison
df_make = filtered['make'].astype(str)
df_topic = filtered['assigned_topic'].astype(str)
if selected_makes:
    filtered = filtered[df_make.isin(selected_makes)]
if selected_topic:
    filtered = filtered[df_topic.isin(selected_topic)]

# Keep only positive sentiment
filtered = filtered[filtered['sentiment']=='positive']

# Rank by combined metrics
filtered['rank_score'] = (
    0.4*filtered['make_cluster_perc'] +
    0.4*filtered['assigned_topic_cluster_perc'] +
    0.2*filtered['sentiment_score']
)
# Unique make-model
filtered = filtered.drop_duplicates(subset=['make','model'])

top3 = filtered.nlargest(3, 'rank_score')

# --- Few-shot examples ---
FEW_SHOT = [
    ("2025 Mercedes SL: Smooth ride, powerful V8...", "The 2025 Mercedes-Benz SL remains a benchmark..."),
    ("2025 Toyota Camry: Reliable sedan...", "The 2025 Toyota Camry delivers class-leading reliability...")
]

# --- Recommendation Display ---
if st.button('Recommend'):
    if top3.empty:
        st.warning('No recommendations found.')
    else:
        for _, car in top3.iterrows():
            title = car['vehicle_title']
            stars = '⭐' * int(round(car['rating']))
            # Build prompt
            messages = [
                {'role':'system','content':'You are a friendly automotive reviewer.'},
                {'role':'user','content':
                    f"Write a friendly, step-by-step review of {title}, highlighting {car['text_for_clustering']}. "
                    f"Mention its strengths ({car['strengths']}) and cons ({car['weaknesses']}). "
                    f"Note the rating ({car['rating']}/5) and sentiment score ({car['sentiment_score']:.2f})."
                    f"Assume that the reader has not purchased his or her vehicle yet and you help him/her decide. So avoid saying Congratulations on your new car. "
                    f"Use different greetings in every car review. Make sure that the beginning of the review is different for each recommended car. Do not use the same greeting for every car review. "
                }
            ]
            resp = client.chat.completions.create(
                model='gpt-3.5-turbo',
                messages=messages,
                temperature=0.7
            )
            summary = resp.choices[0].message.content.strip()
            # Display
            st.markdown(
                f"<div style='background-color:rgba(255,255,255,0.9);padding:15px;border-radius:10px;margin-bottom:15px;'>"
                f"<h3>{title} {stars}</h3>"
                f"<p>{summary}</p>"
                f"<p><a href=\"https://www.edmunds.com/{car['make'].lower()}/\" target='_blank'>&#128279; More info</a></p>"
                f"</div>", unsafe_allow_html=True
            )

