import streamlit as st
from app.agent.decision_maker import generate_answer

st.set_page_config(page_title="RAG Research Assistant", page_icon="📚", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@500;600&display=swap');
.stApp { background: linear-gradient(135deg, #081525 0%, #10243d 55%, #142e4c 100%); color: #eaf2fb; font-family: 'DM Sans', sans-serif; }
.block-container { max-width: 1050px; padding-top: 3rem; padding-bottom: 3rem; }
header, footer { visibility: hidden; }
.hero { margin-top: 55px; margin-bottom: 65px; }
.hero h1 { font-family: 'Playfair Display', serif; font-size: clamp(56px, 9vw, 104px); font-weight: 500; letter-spacing: -4px; line-height: 1.1; color: #edf5ff; margin: 0; }
.hero p { font-size: 19px; color: #9bb5d1; margin-top: 12px; }
.section-label { color: #8eafd2; font-size: 12px; font-weight: 700; letter-spacing: 2.5px; margin-bottom: 18px; }
div.stButton > button { background: #183653; color: #dcecff; border: 1px solid #345778; border-radius: 12px; padding: 13px 24px; font-weight: 600; }
div.stButton > button:hover { background: #244d73; border-color: #75a9d7; color: white; }
div[data-baseweb="input"], div[data-baseweb="textarea"] { background: #152b43; border: 1px solid #365574; border-radius: 16px; }
div[data-baseweb="input"] input, div[data-baseweb="textarea"] textarea { color: #edf5ff !important; background: transparent !important; }
div[data-testid="stMarkdownContainer"] p { color: #b5c8dc; }
hr { border-color: #29435e; }
.stAlert { background: #152b43; color: #dcecff; border: 1px solid #365574; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>RESEARCH.</h1>
  <p>Your RAG-powered assistant for Agentic RAG research.</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-label">01 — ASK YOUR QUESTION</div>', unsafe_allow_html=True)
question = st.text_area(
    "Your question",
    placeholder="e.g. How does Agentic RAG differ from traditional RAG?",
    label_visibility="collapsed",
    height=110,
)

k = st.slider("Number of retrieved passages", min_value=1, max_value=10, value=5)
st.caption("Ask about Agentic RAG architectures, planning, retrieval, reflection, evaluation, tools, applications or limitations.")

if st.button("Find answer  ↗", type="primary"):
    if not question.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Searching the research knowledge base and generating an answer..."):
            try:
                answer = generate_answer(question.strip(), k=k)
                st.markdown("---")
                st.markdown("### ANSWER")
                st.markdown(answer)
            except Exception as exc:
                st.error(f"Answer generation failed: {exc}")
