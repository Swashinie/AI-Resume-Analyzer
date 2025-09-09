import streamlit as st
import pandas as pd
import torch
import pickle
import PyPDF2
import plotly.graph_objects as go
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# Page Config

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Global Styles (CSS)

def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&family=Poppins:wght@600;700&display=swap');
    :root{
    --bg-start:#0A2540;
    --bg-end:#4C1D95;
    --card:#111827;
    --card-soft:#1f2937;
    --text:#E5E7EB;
    --muted:#9CA3AF;
    --accent:#14B8A6; /* teal */
    --accent-2:#F59E0B; /* amber */
    --accent-3:#10B981; /* green */
    --danger:#EF4444; /* red */
    --info:#60A5FA; /* blue */
    }
    /* Background: soft gradients + light radial accents */
    .stApp{
        background:
        radial-gradient(1200px 800px at 12% -10%, rgba(20,184,166,0.08) 0%, rgba(20,184,166,0) 60%),
        radial-gradient(900px 700px at 95% 10%, rgba(99,102,241,0.12) 0%, rgba(99,102,241,0) 60%),
        linear-gradient(135deg, var(--bg-start) 0%, var(--bg-end) 100%) !important;
        color: var(--text);
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar */
    [data-testid="stSidebar"]{
        background: linear-gradient(180deg, #0B2346 0%, #22104E 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    /* Headings */
    h1,h2,h3,h4{ font-family:'Poppins', sans-serif; color:#FFFFFF; letter-spacing:.2px; }

    /* Card */
    .card{
        background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 16px;
        padding: 18px 18px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.25), inset 0 1px 0 rgba(255,255,255,0.05);
        transition: transform .25s ease, box-shadow .25s ease, border-color .25s ease;
    }
    .card:hover{
        transform: translateY(-4px);
        box-shadow: 0 18px 35px rgba(0,0,0,0.35);
        border-color: rgba(255,255,255,0.18);
    }

    /* Buttons */
    .stButton>button {
        border-radius: 999px;
        background-color: var(--accent);
        color: white;
        border: 1px solid rgba(255,255,255,0.15);
        padding: 10px 20px;
        transition: background-color 0.2s, transform 0.2s, box-shadow 0.2s;
    }
    .stButton>button:hover {
        background-color: #0F766E;
        transform: translateY(-1px);
        box-shadow: 0 8px 20px rgba(20,184,166,0.35);
    }

    /* Chips / Badges */
    .chip{
        display:inline-flex; align-items:center; gap:8px;
        padding:6px 12px; border-radius:999px; font-weight:600; font-size:0.9rem;
        border:1px solid rgba(255,255,255,0.14);
        background: rgba(255,255,255,0.06);
    }
    .chip-strong{ background: rgba(16,185,129,0.15); color:#D1FAE5; border-color: rgba(16,185,129,0.35); }
    .chip-moderate{ background: rgba(245,158,11,0.15); color:#FFEDCC; border-color: rgba(245,158,11,0.35); }
    .chip-weak{ background: rgba(239,68,68,0.15); color:#FFE0E0; border-color: rgba(239,68,68,0.35); }

    .badge{
        padding:6px 10px; border-radius:10px; font-weight:600; font-size:0.85rem; margin: 0 8px 8px 0; display:inline-block;
        border:1px solid rgba(255,255,255,0.12);
        background: rgba(255,255,255,0.06);
    }
    .badge-green{ color:#C7F9E5; border-color: rgba(16,185,129,0.4); }
    .badge-orange{ color:#FFE6BF; border-color: rgba(245,158,11,0.4); }
    .badge-red{ color:#FFD1D1; border-color: rgba(239,68,68,0.4); }

    /* Tooltip */
    .tooltip{ position:relative; display:inline-block; cursor:help; }
    .tooltip .tooltiptext{
        visibility:hidden; width:240px; background:#0b1220; color:#E5E7EB; text-align:left; padding:10px 12px;
        border-radius:8px; border:1px solid rgba(255,255,255,0.10);
        position:absolute; z-index:1; bottom:125%; left:50%; transform:translateX(-50%);
        box-shadow:0 8px 18px rgba(0,0,0,0.35); line-height:1.35;
    }
    .tooltip:hover .tooltiptext{ visibility:visible; }

    /* Actions row */
    .actions{ display:flex; gap:10px; flex-wrap: wrap; }
    .action-btn{
        padding:10px 14px; border-radius:12px; border:1px solid rgba(255,255,255,0.15);
        background: rgba(255,255,255,0.06); color:#E5E7EB; text-decoration:none;
        transition: transform .2s ease, background .2s ease, border-color .2s ease;
    }
    .action-btn:hover{ transform: translateY(-2px); background: rgba(255,255,255,0.10); border-color: rgba(255,255,255,0.25); }

    /* Links */
    a{ color: var(--info); text-decoration: none; }
    a:hover{ text-decoration: underline; }

    /* Mobile: stack columns */
    @media (max-width: 992px){
        .hide-mobile{ display:none !important; }
    }
    </style>
    """, unsafe_allow_html=True)

inject_css()
# Load Models (Cached)

@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    base_bert = AutoModel.from_pretrained('bert-base-uncased')

    class BERTResumeClassifier(torch.nn.Module):
        def __init__(self, n_classes):
            super().__init__()
            self.bert = base_bert
            self.drop = torch.nn.Dropout(0.3)
            self.out = torch.nn.Linear(self.bert.config.hidden_size, n_classes)
        def forward(self, input_ids, attention_mask):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            x = self.drop(outputs.pooler_output)
            return self.out(x)

    classifier = BERTResumeClassifier(n_classes=24)
    state = torch.load("models/best_model.pth", map_location=device)
    classifier.load_state_dict(state)
    classifier.to(device)
    classifier.eval()

    with open("models/label_encoder.pkl","rb") as f:
        label_encoder = pickle.load(f)

    return device, tokenizer, base_bert, classifier, label_encoder

device, tokenizer, base_bert, classifier_model, label_encoder = load_models()


# Helpers (Cached)

@st.cache_data
def embed_text(text: str) -> np.ndarray:
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = base_bert(**tokens)
        cls_emb = outputs.last_hidden_state[:,0,:].numpy()
    return cls_emb

@st.cache_data
def analyze_resume(resume_text: str, job_desc: str):
    # Classification
    t = tokenizer(resume_text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        logits = classifier_model(t['input_ids'], t['attention_mask'])
        pred_idx = int(torch.argmax(logits, dim=1).item())
        confidence = float(torch.softmax(logits, dim=1).max().item())
        category = label_encoder.inverse_transform([pred_idx])[0]

    # Similarity
    r_emb = embed_text(resume_text)
    j_emb = embed_text(job_desc)
    sim = float(cosine_similarity(r_emb, j_emb))

    score = (confidence * 0.4) + (sim * 0.6)
    return {
        "category": category,
        "confidence": confidence,
        "similarity": sim,
        "score": score
    }

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for p in reader.pages:
        try:
            text += p.extract_text() or ""
        except:
            continue
    return text.strip()


# Sidebar Navigation

with st.sidebar:
    st.markdown("## Navigation")
    nav = st.radio("Go to", ["Home","Upload","Reports","Settings"])
    st.markdown("---")
    st.caption("Tip: Drag & drop a PDF resume in the Upload page for instant analysis.")


# Layout: Center + Right
col_main, col_right = st.columns([4, 1])

if nav == "Home":
    with col_main:
        st.markdown("## Smarter Resumes. Faster Hiring.")
        st.markdown("""
        <div class="card">
        <h3>Welcome</h3>
        <p>Upload resumes, compare against job descriptions, and receive ATS scores, keyword matches, strengths, weaknesses, and actionable recommendations. Built for recruiters, designed for speed.</p>
        <div class="actions">
        <a class="action-btn" href="#upload">⬆️ Upload Resume</a>
        <a class="action-btn" href="#reports">📊 View Reports</a>
        </div>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown("### Quick Actions")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("- Download latest report")
        st.markdown("- Share with hiring manager")
        st.markdown("- Settings and preferences")
        st.markdown("</div>", unsafe_allow_html=True)

elif nav == "Upload":
    with col_main:
        st.markdown('<a name="upload"></a>', unsafe_allow_html=True)
        st.markdown("## Upload & Analyze")

        up_col, jd_col = st.columns(2)
        with up_col:
            resume_file = st.file_uploader("Upload Resume (PDF)", type="pdf")
        with jd_col:
            job_desc = st.text_area("Paste Job Description", height=200, placeholder="Paste the job description here...")

        if st.button("Analyze"):
            if not (resume_file and job_desc.strip()):
                st.error("Please upload a resume and provide a job description.")
            else:
                with st.spinner("Analyzing resume..."):
                    resume_text = extract_text_from_pdf(resume_file)
                    result = analyze_resume(resume_text, job_desc)

                # Summary cards
                c1,c2,c3 = st.columns(3)
                with c1:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown("#### Predicted Category")
                    st.markdown(f"**{result['category']}**")
                    st.markdown("</div>", unsafe_allow_html=True)
                with c2:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown("#### Confidence")
                    st.markdown(f"**{result['confidence']*100:.1f}%**")
                    st.markdown("</div>", unsafe_allow_html=True)
                with c3:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown("#### Similarity")
                    st.markdown(f"**{result['similarity']*100:.1f}%**")
                    st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("## Results")

                # Charts row
                gcol, bcol = st.columns(2)
                with gcol:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown("#### ATS Compatibility Score")
                    ats_score = result['score']*100
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=ats_score,
                        title={'text': "ATS Score"},
                        gauge={
                            'axis': {'range':[0,100], 'tickcolor':'#9CA3AF'},
                            'bar': {'color': '#14B8A6'},
                            'bgcolor': 'rgba(255,255,255,0.04)',
                            'borderwidth': 2,
                            'bordercolor': 'rgba(255,255,255,0.15)'
                        },
                        number={'font': {'color':'#E5E7EB'}}
                    ))
                    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color':'#E5E7EB'})
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                with bcol:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown("#### Keyword Match %")
                    kw_val = result['similarity']*100
                    fig2 = go.Figure(go.Bar(
                        x=[kw_val], y=["Match"], orientation="h",
                        marker_color="#F59E0B"
                    ))
                    fig2.update_layout(
                        xaxis=dict(range=[0,100], showgrid=False, color="#E5E7EB"),
                        yaxis=dict(showgrid=False, color="#E5E7EB"),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                # Insights
                st.markdown("### Insights")
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("**Strengths**")
                st.markdown("""
                  <span class="badge badge-green">ATS Friendly</span>
                  <span class="badge badge-green">Relevant Skills</span>
                  <span class="badge badge-green">Clear Structure</span>
                """, unsafe_allow_html=True)

                st.markdown("**Weaknesses**")
                st.markdown("""
                  <span class="tooltip badge badge-red">Missing Role Keywords
                    <span class="tooltiptext">Add role-specific keywords found in JD to improve ATS matching.</span>
                  </span>
                  <span class="tooltip badge badge-red">Sparse Metrics
                    <span class="tooltiptext">Quantify achievements (e.g., increased revenue by 20%).</span>
                  </span>
                """, unsafe_allow_html=True)

                # Recommendation chip
                rec_label, rec_class = ("Strong Fit", "chip-strong") if ats_score>=80 else (("Moderate Fit", "chip-moderate") if ats_score>=60 else ("Needs Improvement", "chip-weak"))
                st.markdown(f"**Recommendation:** <span class='chip {rec_class}'>{rec_label}</span>", unsafe_allow_html=True)

                # Gamification badges
                st.markdown("### Badges")
                st.markdown("""
                  <span class="badge badge-green">Skill-Rich</span>
                  <span class="badge badge-orange">Well-Formatted</span>
                """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown("### Quick Tips")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("- Include exact keywords from the job description.")
        st.write("- Quantify achievements with numbers.")
        st.write("- Keep formatting simple and ATS-friendly.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### Actions")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="actions"><a class="action-btn" href="#">📥 Download PDF</a><a class="action-btn" href="#">🔗 Share</a></div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

elif nav == "Reports":
    with col_main:
        st.markdown('<a name="reports"></a>', unsafe_allow_html=True)
        st.markdown("## Reports")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        df = pd.DataFrame({
            "Candidate":["Alice","Bob","Carol"],
            "Category":["DATA-SCIENCE","ACCOUNTANT","PROJECT-MGMT"],
            "Similarity":["82%","71%","64%"],
            "ATS Score":["85%","76%","61%"],
            "Recommendation":["Strong Fit","Moderate Fit","Needs Improvement"]
        })
        st.dataframe(df, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

elif nav == "Settings":
    with col_main:
        st.markdown("## Settings")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.checkbox("Enable compact mode")
        st.selectbox("Font", ["Inter","Poppins","System Default"])
        st.slider("Global scale", 90, 120, 100, help="Adjusts font sizes and spacing.")
        st.markdown("</div>", unsafe_allow_html=True)
