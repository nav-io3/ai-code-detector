import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from core import features, featurize, fingerprints, containment

HERE = Path(__file__).parent
ART = HERE / "artifacts"          # model.joblib + reference.csv (from the notebook)
BERT_DIR = HERE / "codebert"      # optional

st.set_page_config(page_title="AI Code Detector", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner="Loading model and reference set...")
def load_all():
    bundle = joblib.load(ART / "model.joblib")
    ref = pd.read_csv(ART / "reference.csv")
    ref["fp"] = ref.code.map(fingerprints)
    X = featurize(ref.code.tolist(), bundle["cols"])
    stats = X.groupby(ref.label.values).median()
    return bundle, ref, stats


@st.cache_resource(show_spinner="Loading CodeBERT...")
def load_bert():
    if not BERT_DIR.exists():
        return None
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tok = AutoTokenizer.from_pretrained(str(BERT_DIR))
        model = AutoModelForSequenceClassification.from_pretrained(str(BERT_DIR)).eval()
        return tok, model, torch
    except Exception:
        return None


def bert_prob(code):
    b = load_bert()
    if b is None:
        return None
    tok, model, torch = b
    enc = tok(code, truncation=True, max_length=384, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits
    return float(torch.softmax(logits, -1)[0, 1])


def evidence(bundle, x, stats, k=8):
    imp = pd.Series(bundle["rf"].feature_importances_, index=bundle["cols"]).sort_values(ascending=False).head(k)
    rows = []
    for f in imp.index:
        v, h, a = float(x[f].iloc[0]), float(stats.loc[0, f]), float(stats.loc[1, f])
        if abs(h - a) < 1e-9:
            lean = "neutral"
        else:
            lean = "closer to AI" if abs(v - a) < abs(v - h) else "closer to human"
        rows.append({"feature": f, "this code": round(v, 3), "human median": round(h, 3),
                     "AI median": round(a, 3), "leans": lean})
    return pd.DataFrame(rows)


def band(p):
    if p < 0.35:
        return "Low AI-likelihood"
    if p < 0.65:
        return "Uncertain - no strong signal either way"
    return "High AI-likelihood - review manually"


def check_syntax(code):
    try:
        compile(code, "<submission>", "exec")
        return None
    except SyntaxError as e:
        return f"Syntax error on line {e.lineno}: {e.msg}"


bundle, ref, stats = load_all()

st.title("AI-Generated Code Detector + Similarity Checker")
st.warning("This score is a statistical signal for a teacher to review. It is NOT proof of cheating. "
           "Detectors can be wrong, especially on edited code, newer models, or students who write clean textbook-style code.")

tab1, tab2 = st.tabs(["Single submission", "Compare submissions (batch)"])

with tab1:
    left, right = st.columns([3, 2])
    with left:
        up = st.file_uploader("Upload a .py file", type=["py"], key="single_up")
        pasted = st.text_area("...or paste code", height=320, key="single_text")
        code = up.read().decode("utf-8", errors="ignore") if up else pasted
        go = st.button("Analyse", type="primary")
    with right:
        thr = st.slider("Similarity flag threshold", 0.3, 1.0, 0.6, 0.05, key="thr1")

    if go:
        if not code.strip():
            st.error("Please paste or upload some code.")
        elif (err := check_syntax(code)):
            st.error(err + ". This detector needs valid Python.")
        else:
            x = featurize([code], bundle["cols"])
            p = float(bundle["xgb"].predict_proba(x)[0, 1])
            c1, c2 = st.columns(2)
            c1.metric("AI-likelihood (feature model)", f"{p * 100:.0f}%")
            c1.progress(min(max(p, 0.0), 1.0))
            c1.caption(band(p))
            pb = bert_prob(code)
            if pb is not None:
                c2.metric("AI-likelihood (CodeBERT)", f"{pb * 100:.0f}%")
                c2.progress(min(max(pb, 0.0), 1.0))
            else:
                c2.caption("CodeBERT not loaded (put the unzipped `codebert/` folder next to app.py to enable it).")

            st.subheader("Evidence")
            st.caption("Top features by importance, compared with the medians of known human and AI code in the reference set.")
            st.dataframe(evidence(bundle, x, stats), width="stretch", hide_index=True)

            st.subheader("Similarity against reference set")
            fp = fingerprints(code)
            sims = ref.fp.map(lambda f: containment(fp, f))
            top = ref.assign(similarity=sims).sort_values("similarity", ascending=False).head(5)
            out = pd.DataFrame({"reference problem": top.problem_id, "reference is AI": top.label.map({0: "no", 1: "yes"}),
                                "similarity %": (top.similarity * 100).round(1),
                                "flag": np.where(top.similarity >= thr, "NEAR-IDENTICAL", "")})
            st.dataframe(out, width="stretch", hide_index=True)
            if (top.similarity >= thr).any():
                st.error("Near-identical structure found in the reference set. Review the matching programs.")

with tab2:
    st.write("Upload several submissions for the same assignment to find pairs with near-identical structure.")
    files = st.file_uploader("Upload .py files", type=["py"], accept_multiple_files=True, key="batch_up")
    thr2 = st.slider("Flag pairs at or above", 0.3, 1.0, 0.6, 0.05, key="thr2")
    if files and len(files) >= 2:
        subs = {f.name: f.read().decode("utf-8", errors="ignore") for f in files}
        valid = {n: c for n, c in subs.items() if not check_syntax(c)}
        for n in set(subs) - set(valid):
            st.warning(f"Skipped {n}: syntax error")
        names = list(valid)
        fps = {n: fingerprints(valid[n]) for n in names}
        pairs = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                s = containment(fps[names[i]], fps[names[j]])
                pairs.append({"A": names[i], "B": names[j], "similarity %": round(s * 100, 1),
                              "flag": "NEAR-IDENTICAL" if s >= thr2 else ""})
        st.subheader("Pairwise similarity")
        st.dataframe(pd.DataFrame(pairs).sort_values("similarity %", ascending=False),
                     width="stretch", hide_index=True)
        st.subheader("AI-likelihood per file")
        X = featurize([valid[n] for n in names], bundle["cols"])
        probs = bundle["xgb"].predict_proba(X)[:, 1]
        st.dataframe(pd.DataFrame({"file": names, "AI-likelihood %": (probs * 100).round(0), "reading": [band(p) for p in probs]})
                     .sort_values("AI-likelihood %", ascending=False), width="stretch", hide_index=True)
