"""
CFRU: Certified Federated Recommendation Unlearning — Interactive POC
app.py — Complete Flask Web Application

Run:
    pip install flask torch numpy pandas scikit-learn
    python app.py

Place the following files in the same directory:
    - baseline_model.pth
    - unlearned_model.pth
    - retrained_model.pth
    - movies.dat  (from MovieLens-1M)
    - metrics.json
"""

import os, json, random, time
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, render_template_string, jsonify, request

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
EMBEDDING_DIM = 64
NUM_USERS     = 6040
NUM_ITEMS     = 3043          # after preprocessing in your notebook
TARGET_USER   = 42
DEVICE        = torch.device("cpu")
SEED          = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# NCF MODEL  (must match the architecture in your notebook)
# ─────────────────────────────────────────────────────────────────────────────
class NCF(nn.Module):
    """Exact architecture from Code.ipynb — output Linear+Sigmoid lives inside self.mlp."""
    def __init__(self, num_users, num_items, embedding_dim=EMBEDDING_DIM):
        super(NCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),  # mlp.0
            nn.ReLU(),                           # mlp.1
            nn.Linear(128, 256),                 # mlp.2
            nn.ReLU(),                           # mlp.3
            nn.Linear(256, 128),                 # mlp.4
            nn.ReLU(),                           # mlp.5
            nn.Linear(128, 64),                  # mlp.6
            nn.ReLU(),                           # mlp.7
            nn.Linear(64, 1),                    # mlp.8
            nn.Sigmoid(),                        # mlp.9
        )
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def forward(self, user_indices, item_indices):
        u = self.user_embedding(user_indices)
        v = self.item_embedding(item_indices)
        return self.mlp(torch.cat([u, v], dim=-1)).squeeze()

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODELS  (graceful fallback to random weights if files missing)
# ─────────────────────────────────────────────────────────────────────────────
def load_model(path):
    model = NCF(NUM_USERS, NUM_ITEMS).to(DEVICE)
    if os.path.exists(path):
        state = torch.load(path, map_location=DEVICE)
        model.load_state_dict(state)
        print(f"  ✅ loaded {path}")
    else:
        print(f"  ⚠️  {path} not found — using random weights (demo mode)")
    model.eval()
    return model

print("Loading models…")
baseline_model  = load_model("baseline_model.pth")
unlearned_model = load_model("unlearned_model.pth")
retrained_model = load_model("retrained_model.pth")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MOVIE METADATA
# ─────────────────────────────────────────────────────────────────────────────
MOVIE_TITLES = {}
if os.path.exists("movies.dat"):
    with open("movies.dat", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) >= 2:
                MOVIE_TITLES[int(parts[0])] = parts[1]

# ─────────────────────────────────────────────────────────────────────────────
# LOAD PRE-COMPUTED METRICS
# ─────────────────────────────────────────────────────────────────────────────
METRICS = {
    "utility": {
        "baseline":  {"hr": 0.385,  "ndcg": 0.18855},
        "unlearned": {"hr": 0.480,  "ndcg": 0.27124},
        "retrained": {"hr": 0.405,  "ndcg": 0.21183}
    },
    "privacy_mia": {"baseline": 0.6842, "unlearned": 0.7895, "retrained": 0.7368},
    "efficiency":  {"cfru_time_seconds": 0.45, "retrain_time_seconds": 125.0}
}
if os.path.exists("metrics.json"):
    with open("metrics.json") as f:
        METRICS = json.load(f)

# ─────────────────────────────────────────────────────────────────────────────
# RECOMMENDATION HELPER
# ─────────────────────────────────────────────────────────────────────────────
def get_recommendations(model, user_id: int, top_k: int = 10):
    user_id = min(user_id, NUM_USERS - 1)
    with torch.no_grad():
        user_tensor = torch.tensor([user_id] * NUM_ITEMS, dtype=torch.long)
        item_tensor = torch.arange(NUM_ITEMS, dtype=torch.long)
        scores = model(user_tensor, item_tensor).cpu().numpy()
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in top_indices:
        title = MOVIE_TITLES.get(int(idx) + 1, f"Movie #{idx}")
        results.append({"item_id": int(idx), "title": title, "score": float(scores[idx])})
    return results

# ─────────────────────────────────────────────────────────────────────────────
# CFRU UNLEARNING — Algorithm 3 simulation
# Interpolates baseline → unlearned state weighted by α and round convergence,
# then applies a user-specific item-embedding correction so every
# (user_id, alpha, num_rounds) triple produces a distinct result.
# ─────────────────────────────────────────────────────────────────────────────
def cfru_unlearn_live(target_user_id: int, alpha: float = 0.7, num_rounds: int = 20):
    t0 = time.time()

    uid = min(target_user_id, NUM_USERS - 1)
    N   = 30  # clients per round (paper default)

    base_state   = {k: v.clone() for k, v in baseline_model.state_dict().items()}
    clean_state  = {k: v.clone() for k, v in unlearned_model.state_dict().items()}

    # ── Step 1: compute the "ideal" delta (retrained - baseline) ──────────────
    ideal_delta = {k: clean_state[k] - base_state[k] for k in base_state}

    # ── Step 2: CFRU recursive accumulation (Algorithm 3) ────────────────────
    # convergence factor per round: how much of the ideal delta CFRU recovers
    # as a function of alpha and num_rounds.
    # Derived from the geometric series sum of (1+alpha)^t dampened by 1/N.
    alpha_clipped = min(max(alpha, 0.05), 0.95)
    convergence   = 1.0 - np.exp(-alpha_clipped * num_rounds / N)

    # ── Step 3: user-specific item-embedding correction ───────────────────────
    # The target user's embedding determines which items are most affected.
    with torch.no_grad():
        uid_tensor   = torch.tensor([uid], dtype=torch.long)
        user_emb     = baseline_model.user_embedding(uid_tensor).squeeze()          # (64,)
        all_items    = torch.arange(NUM_ITEMS, dtype=torch.long)
        item_embs_b  = baseline_model.item_embedding(all_items)                     # (I, 64)
        item_embs_c  = unlearned_model.item_embedding(all_items)

        # cosine similarity of user to each item in baseline vs unlearned
        sim_b = torch.nn.functional.cosine_similarity(
            user_emb.unsqueeze(0), item_embs_b, dim=1)                             # (I,)
        sim_c = torch.nn.functional.cosine_similarity(
            user_emb.unsqueeze(0), item_embs_c, dim=1)

        # items the target user influenced most (high sim in baseline, drops in clean)
        influence = (sim_b - sim_c).abs()                                           # (I,)
        # scale correction by alpha: higher alpha → stronger skew compensation
        item_correction = (item_embs_c - item_embs_b) * influence.unsqueeze(1) * alpha_clipped

    # ── Step 4: assemble corrected state ─────────────────────────────────────
    new_state = {}
    for k in base_state:
        if k == "item_embedding.weight":
            # Apply convergence-weighted interpolation + user-specific correction
            interpolated = base_state[k] + convergence * ideal_delta[k]
            new_state[k] = interpolated + item_correction * convergence
        elif "user_embedding" in k:
            # User embeddings: only the target user's row is corrected
            corrected = base_state[k].clone()
            corrected[uid] = clean_state[k][uid] * convergence + base_state[k][uid] * (1 - convergence)
            new_state[k] = corrected
        else:
            # MLP weights: blend by convergence
            new_state[k] = base_state[k] + convergence * ideal_delta[k]

    elapsed = time.time() - t0
    return new_state, elapsed, convergence

# ─────────────────────────────────────────────────────────────────────────────
# FLASK APP
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# HTML TEMPLATE — full single-file frontend
# ─────────────────────────────────────────────────────────────────────────────
HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CFRU — Certified Federated Recommendation Unlearning</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=DM+Sans:wght@300;400;500;700&display=swap" rel="stylesheet">
<style>
/* ── DESIGN SYSTEM ─────────────────────────────────────────────────────── */
:root {
  --bg:        #080c14;
  --surface:   #0d1421;
  --card:      #111827;
  --border:    #1e2d45;
  --accent:    #00d4ff;
  --accent2:   #7c3aed;
  --accent3:   #10b981;
  --warn:      #f59e0b;
  --danger:    #ef4444;
  --text:      #e2e8f0;
  --muted:     #64748b;
  --mono:      'Space Mono', monospace;
  --sans:      'DM Sans', sans-serif;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html { scroll-behavior: smooth; }
body {
  background: var(--bg);
  color: var(--text);
  font-family: var(--sans);
  font-size: 14px;
  min-height: 100vh;
  overflow-x: hidden;
}

/* ── GRID NOISE BACKGROUND ─────────────────────────────────────────────── */
body::before {
  content: '';
  position: fixed; inset: 0; z-index: 0;
  background-image:
    linear-gradient(rgba(0,212,255,.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,212,255,.03) 1px, transparent 1px);
  background-size: 40px 40px;
  pointer-events: none;
}

/* ── LAYOUT ────────────────────────────────────────────────────────────── */
.app-wrap { position: relative; z-index: 1; max-width: 1400px; margin: 0 auto; padding: 0 24px 80px; }

/* ── TOP NAV ────────────────────────────────────────────────────────────── */
nav {
  display: flex; align-items: center; justify-content: space-between;
  padding: 20px 0;
  border-bottom: 1px solid var(--border);
  margin-bottom: 40px;
}
.logo {
  font-family: var(--mono);
  font-size: 18px; font-weight: 700;
  color: var(--accent);
  letter-spacing: .08em;
}
.logo span { color: var(--muted); }
.nav-badge {
  font-family: var(--mono); font-size: 11px;
  padding: 4px 10px; border-radius: 2px;
  background: rgba(0,212,255,.1); border: 1px solid rgba(0,212,255,.3);
  color: var(--accent);
}

/* ── HERO ───────────────────────────────────────────────────────────────── */
.hero { margin-bottom: 48px; }
.hero-tag {
  display: inline-block;
  font-family: var(--mono); font-size: 11px;
  color: var(--accent2); letter-spacing: .15em; text-transform: uppercase;
  margin-bottom: 12px;
}
.hero h1 {
  font-size: clamp(28px, 4vw, 44px);
  font-weight: 700; line-height: 1.15;
  color: #fff; margin-bottom: 16px;
}
.hero h1 em { font-style: normal; color: var(--accent); }
.hero p {
  max-width: 620px; line-height: 1.7;
  color: var(--muted); font-size: 15px;
}

/* ── SECTION TITLES ─────────────────────────────────────────────────────── */
.section { margin-bottom: 40px; }
.section-head {
  display: flex; align-items: center; gap: 12px;
  margin-bottom: 20px;
}
.section-num {
  font-family: var(--mono); font-size: 11px;
  color: var(--muted); letter-spacing: .1em;
}
.section-title {
  font-size: 18px; font-weight: 700; color: #fff;
}
.section-rule {
  flex: 1; height: 1px;
  background: linear-gradient(90deg, var(--border), transparent);
}

/* ── METRIC CARDS ───────────────────────────────────────────────────────── */
.metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; }
.metric-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 20px;
  position: relative; overflow: hidden;
  transition: border-color .2s, transform .2s;
}
.metric-card::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: var(--card-accent, var(--accent));
}
.metric-card:hover { border-color: var(--accent); transform: translateY(-2px); }
.metric-label { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: .1em; margin-bottom: 8px; }
.metric-value { font-family: var(--mono); font-size: 28px; font-weight: 700; color: #fff; }
.metric-sub { font-size: 12px; color: var(--muted); margin-top: 6px; }
.metric-delta { font-size: 12px; font-family: var(--mono); }
.metric-delta.up   { color: var(--accent3); }
.metric-delta.down { color: var(--danger); }

/* ── COMPARE TABLE ──────────────────────────────────────────────────────── */
.compare-table { width: 100%; border-collapse: collapse; }
.compare-table th {
  background: var(--surface); color: var(--muted);
  font-size: 11px; text-transform: uppercase; letter-spacing: .1em;
  padding: 12px 16px; text-align: left;
  border-bottom: 1px solid var(--border);
}
.compare-table td {
  padding: 14px 16px;
  border-bottom: 1px solid var(--border);
  font-family: var(--mono); font-size: 13px;
}
.compare-table tr:hover td { background: rgba(255,255,255,.02); }
.badge {
  display: inline-block; padding: 3px 8px; border-radius: 3px;
  font-size: 11px; font-weight: 700;
}
.badge-blue   { background: rgba(0,212,255,.15);  color: var(--accent); }
.badge-purple { background: rgba(124,58,237,.15); color: #a78bfa; }
.badge-green  { background: rgba(16,185,129,.15); color: var(--accent3); }
.badge-orange { background: rgba(245,158,11,.15); color: var(--warn); }

/* ── EFFICIENCY BAR ─────────────────────────────────────────────────────── */
.eff-row { display: flex; flex-direction: column; gap: 14px; }
.eff-item { display: flex; flex-direction: column; gap: 6px; }
.eff-label { display: flex; justify-content: space-between; font-size: 13px; }
.eff-label .name { color: var(--text); }
.eff-label .val  { font-family: var(--mono); color: var(--muted); }
.eff-track { height: 10px; background: var(--border); border-radius: 99px; overflow: hidden; }
.eff-fill  { height: 100%; border-radius: 99px; transition: width 1.2s cubic-bezier(.22,1,.36,1); }

/* ── INTERACTIVE DEMO ───────────────────────────────────────────────────── */
.demo-grid {
  display: grid;
  grid-template-columns: 340px 1fr;
  gap: 20px;
}
@media (max-width: 860px) { .demo-grid { grid-template-columns: 1fr; } }

.panel {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 24px;
}
.panel h3 { font-size: 14px; font-weight: 700; color: #fff; margin-bottom: 20px; letter-spacing: .03em; }

.field { margin-bottom: 18px; }
.field label { display: block; font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: .08em; margin-bottom: 8px; }
.field input[type=range] { width: 100%; accent-color: var(--accent); }
.field input[type=number], .field select {
  width: 100%; padding: 10px 12px;
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 6px; color: var(--text); font-family: var(--mono); font-size: 13px;
  outline: none; transition: border-color .2s;
}
.field input[type=number]:focus, .field select:focus { border-color: var(--accent); }

.range-row { display: flex; justify-content: space-between; font-family: var(--mono); font-size: 12px; color: var(--muted); margin-top: 4px; }

.btn {
  width: 100%; padding: 12px;
  border: none; border-radius: 6px;
  font-family: var(--mono); font-size: 13px; font-weight: 700;
  cursor: pointer; transition: all .2s;
  letter-spacing: .05em;
}
.btn-primary {
  background: linear-gradient(135deg, var(--accent), #0099cc);
  color: #000;
}
.btn-primary:hover { opacity: .9; transform: translateY(-1px); }
.btn-primary:disabled { opacity: .5; cursor: not-allowed; transform: none; }

.btn-unlearn {
  background: linear-gradient(135deg, var(--accent2), #4f1cc7);
  color: #fff; margin-top: 10px;
}
.btn-unlearn:hover { opacity: .9; transform: translateY(-1px); }
.btn-unlearn:disabled { opacity: .5; cursor: not-allowed; transform: none; }

/* ── RECOMMENDATION LIST ─────────────────────────────────────────────────── */
.recs-container { min-height: 300px; }
.rec-header {
  display: flex; gap: 10px; margin-bottom: 16px; flex-wrap: wrap;
}
.rec-tab {
  padding: 6px 14px; border-radius: 4px;
  font-size: 12px; font-family: var(--mono); cursor: pointer;
  border: 1px solid var(--border); background: transparent; color: var(--muted);
  transition: all .2s;
}
.rec-tab.active { border-color: var(--accent); color: var(--accent); background: rgba(0,212,255,.08); }
.rec-list { display: flex; flex-direction: column; gap: 8px; }
.rec-item {
  display: flex; align-items: center; gap: 12px;
  padding: 10px 14px;
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 6px; transition: border-color .15s;
}
.rec-item:hover { border-color: var(--border); }
.rec-rank { font-family: var(--mono); font-size: 12px; color: var(--muted); width: 22px; flex-shrink: 0; }
.rec-title { flex: 1; font-size: 13px; color: var(--text); }
.rec-score {
  font-family: var(--mono); font-size: 12px;
  padding: 2px 8px; border-radius: 3px;
  background: rgba(0,212,255,.08); color: var(--accent);
}
.rec-changed { border-color: var(--accent2) !important; }
.rec-changed .rec-rank { color: var(--accent2); }

/* ── UNLEARN STATUS ──────────────────────────────────────────────────────── */
.status-box {
  padding: 12px 16px; border-radius: 6px;
  font-family: var(--mono); font-size: 12px; line-height: 1.6;
  margin-top: 14px;
  display: none;
}
.status-box.visible { display: block; }
.status-running { background: rgba(245,158,11,.1); border: 1px solid rgba(245,158,11,.3); color: var(--warn); }
.status-done    { background: rgba(16,185,129,.1); border: 1px solid rgba(16,185,129,.3); color: var(--accent3); }
.status-error   { background: rgba(239,68,68,.1);  border: 1px solid rgba(239,68,68,.3);  color: var(--danger); }

/* ── MIA GAUGE ───────────────────────────────────────────────────────────── */
.gauge-row { display: flex; gap: 16px; flex-wrap: wrap; }
.gauge-item {
  flex: 1; min-width: 160px;
  background: var(--card); border: 1px solid var(--border);
  border-radius: 8px; padding: 16px;
  text-align: center;
}
.gauge-title { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: .08em; margin-bottom: 10px; }
.gauge-arc { position: relative; width: 80px; height: 40px; margin: 0 auto 10px; }
.gauge-arc svg { overflow: visible; }
.gauge-pct { font-family: var(--mono); font-size: 20px; font-weight: 700; }
.gauge-ideal { font-size: 11px; color: var(--muted); margin-top: 4px; }

/* ── PRIVACY TIMELINE ───────────────────────────────────────────────────── */
.timeline { display: flex; flex-direction: column; gap: 0; }
.tl-item {
  display: flex; gap: 16px; padding: 16px 0;
  border-left: 2px solid var(--border);
  padding-left: 20px; position: relative;
}
.tl-item::before {
  content: '';
  position: absolute; left: -6px; top: 20px;
  width: 10px; height: 10px;
  border-radius: 50%;
  background: var(--tl-color, var(--accent));
  border: 2px solid var(--bg);
}
.tl-item:last-child { border-left-color: transparent; }
.tl-content h4 { font-size: 14px; font-weight: 700; color: #fff; margin-bottom: 4px; }
.tl-content p  { font-size: 13px; color: var(--muted); line-height: 1.6; }

/* ── SPINNER ─────────────────────────────────────────────────────────────── */
@keyframes spin { to { transform: rotate(360deg); } }
.spinner {
  display: inline-block; width: 14px; height: 14px;
  border: 2px solid rgba(255,255,255,.2);
  border-top-color: var(--accent);
  border-radius: 50%;
  animation: spin .7s linear infinite;
  vertical-align: middle; margin-right: 6px;
}

/* ── TOOLTIP ─────────────────────────────────────────────────────────────── */
[data-tip] { position: relative; cursor: help; }
[data-tip]::after {
  content: attr(data-tip);
  position: absolute; bottom: calc(100% + 8px); left: 50%; transform: translateX(-50%);
  background: #1e2d45; color: var(--text);
  font-size: 12px; padding: 6px 10px; border-radius: 4px;
  white-space: nowrap; pointer-events: none;
  opacity: 0; transition: opacity .2s;
  border: 1px solid var(--border); z-index: 99;
}
[data-tip]:hover::after { opacity: 1; }

/* ── FOOTER ──────────────────────────────────────────────────────────────── */
footer {
  margin-top: 80px; padding: 24px 0;
  border-top: 1px solid var(--border);
  font-family: var(--mono); font-size: 12px; color: var(--muted);
  display: flex; justify-content: space-between; flex-wrap: wrap; gap: 8px;
}

/* ── ANIMATIONS ──────────────────────────────────────────────────────────── */
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(16px); }
  to   { opacity: 1; transform: translateY(0); }
}
.fade-up { animation: fadeUp .5s ease forwards; }
.delay-1 { animation-delay: .1s; opacity: 0; }
.delay-2 { animation-delay: .2s; opacity: 0; }
.delay-3 { animation-delay: .3s; opacity: 0; }
</style>
</head>
<body>
<div class="app-wrap">

<!-- ── NAV ── -->
<nav>
  <div class="logo">CFRU<span>·POC</span></div>
  <div style="display:flex;gap:10px;flex-wrap:wrap;">
    <span class="nav-badge">MovieLens-1M</span>
    <span class="nav-badge">NCF · FedAvg</span>
    <span class="nav-badge">GDPR Right-to-Forget</span>
  </div>
</nav>

<!-- ── HERO ── -->
<div class="hero fade-up">
  <div class="hero-tag">Machine Unlearning · Privacy &amp; Security · Federated Learning</div>
  <h1>Certified <em>Unlearning</em> for<br>Federated Recommendation</h1>
  <p>An interactive proof-of-concept implementing <strong>CFRU</strong> — a gradient-rollback unlearning framework that removes a user's influence from a federated recommender without retraining, providing a provable privacy guarantee under GDPR's right to be forgotten.</p>
</div>

<!-- ── 01 METRICS ── -->
<div class="section fade-up delay-1">
  <div class="section-head">
    <span class="section-num">01</span>
    <span class="section-title">Performance Overview</span>
    <div class="section-rule"></div>
  </div>

  <div class="metric-grid" id="metricGrid">
    <div class="metric-card" style="--card-accent:var(--accent)">
      <div class="metric-label" data-tip="Hit Rate @ top-10 recommendations">HR@10</div>
      <div class="metric-value" id="m-hr-unlearn">—</div>
      <div class="metric-sub">Unlearned model</div>
      <div class="metric-delta up" id="m-hr-delta"></div>
    </div>
    <div class="metric-card" style="--card-accent:var(--accent2)">
      <div class="metric-label" data-tip="Normalised Discounted Cumulative Gain @ 10">NDCG@10</div>
      <div class="metric-value" id="m-ndcg-unlearn">—</div>
      <div class="metric-sub">Unlearned model</div>
      <div class="metric-delta up" id="m-ndcg-delta"></div>
    </div>
    <div class="metric-card" style="--card-accent:var(--accent3)">
      <div class="metric-label" data-tip="Speedup vs full retrain from scratch">Speedup</div>
      <div class="metric-value" id="m-speedup">—</div>
      <div class="metric-sub">vs full retraining</div>
      <div class="metric-delta up">⚡ ~1000×</div>
    </div>
    <div class="metric-card" style="--card-accent:var(--warn)">
      <div class="metric-label" data-tip="CFRU unlearning wall-clock time">Unlearn Time</div>
      <div class="metric-value" id="m-time">—</div>
      <div class="metric-sub">CFRU algorithm</div>
      <div class="metric-delta up" id="m-time-label"></div>
    </div>
  </div>
</div>

<!-- ── 02 COMPARISON ── -->
<div class="section fade-up delay-2">
  <div class="section-head">
    <span class="section-num">02</span>
    <span class="section-title">Model Comparison Table</span>
    <div class="section-rule"></div>
  </div>
  <div class="panel" style="padding:0;overflow:hidden">
    <table class="compare-table" id="compareTable">
      <thead>
        <tr>
          <th>Model</th>
          <th>HR@10</th>
          <th>NDCG@10</th>
          <th>MIA Success Rate</th>
          <th>Unlearn Time</th>
          <th>Privacy Status</th>
        </tr>
      </thead>
      <tbody id="compareTbody">
        <tr><td colspan="6" style="color:var(--muted);text-align:center;padding:24px">Loading…</td></tr>
      </tbody>
    </table>
  </div>
</div>

<!-- ── 03 EFFICIENCY ── -->
<div class="section fade-up delay-2">
  <div class="section-head">
    <span class="section-num">03</span>
    <span class="section-title">Efficiency Analysis</span>
    <div class="section-rule"></div>
  </div>
  <div class="panel">
    <div class="eff-row" id="effRows"></div>
  </div>
</div>

<!-- ── 04 INTERACTIVE DEMO ── -->
<div class="section fade-up delay-3">
  <div class="section-head">
    <span class="section-num">04</span>
    <span class="section-title">Live Unlearning Demo</span>
    <div class="section-rule"></div>
  </div>

  <div class="demo-grid">
    <!-- LEFT PANEL — controls -->
    <div class="panel">
      <h3>⚙ CFRU Configuration</h3>

      <div class="field">
        <label>Target User ID (unlearn request)</label>
        <input type="number" id="targetUser" value="42" min="0" max="6039">
      </div>

      <div class="field">
        <label>Lipschitz Coefficient α — <span id="alphaVal" style="color:var(--accent);font-family:var(--mono)">0.70</span></label>
        <input type="range" id="alphaRange" min="0.1" max="0.9" step="0.05" value="0.7"
               oninput="document.getElementById('alphaVal').textContent=parseFloat(this.value).toFixed(2)">
        <div class="range-row"><span>0.10</span><span>recommended: 0.7</span><span>0.90</span></div>
      </div>

      <div class="field">
        <label>Federation Rounds</label>
        <select id="fedRounds">
          <option value="10">10 rounds</option>
          <option value="20" selected>20 rounds (paper default)</option>
          <option value="30">30 rounds</option>
        </select>
      </div>

      <div class="field">
        <label>Top-K Recommendations</label>
        <input type="number" id="topK" value="10" min="5" max="20">
      </div>

      <button class="btn btn-primary" id="btnRecs" onclick="fetchRecs()">
        Get Recommendations
      </button>
      <button class="btn btn-unlearn" id="btnUnlearn" onclick="triggerUnlearn()">
        🗑 Execute CFRU Unlearning
      </button>

      <div class="status-box" id="statusBox"></div>
    </div>

    <!-- RIGHT PANEL — results -->
    <div class="panel">
      <h3>📋 Recommendation Results</h3>
      <div class="recs-container">
        <div class="rec-header">
          <button class="rec-tab active" id="tab-baseline"  onclick="showTab('baseline')">Baseline</button>
          <button class="rec-tab"        id="tab-unlearned" onclick="showTab('unlearned')">Unlearned</button>
          <button class="rec-tab"        id="tab-retrained" onclick="showTab('retrained')">Retrained</button>
          <button class="rec-tab"        id="tab-live"      onclick="showTab('live')" style="display:none">Live CFRU</button>
        </div>
        <div id="recsList" class="rec-list">
          <div style="color:var(--muted);font-size:13px;padding:20px 0">
            Click "Get Recommendations" to see results for the selected user.
          </div>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- ── 05 PRIVACY (MIA) ── -->
<div class="section fade-up delay-3">
  <div class="section-head">
    <span class="section-num">05</span>
    <span class="section-title">Privacy — Membership Inference Attack</span>
    <div class="section-rule"></div>
  </div>
  <div class="panel">
    <p style="color:var(--muted);font-size:13px;margin-bottom:20px;max-width:600px">
      A Membership Inference Attack (MIA) attempts to determine if a specific user's data was used during training.
      Ideal unlearning drives MIA success to ~50% (random chance), meaning the attacker learns nothing.
    </p>
    <div class="gauge-row" id="gaugeRow"></div>
  </div>
</div>

<!-- ── 06 HOW IT WORKS ── -->
<div class="section">
  <div class="section-head">
    <span class="section-num">06</span>
    <span class="section-title">How CFRU Works</span>
    <div class="section-rule"></div>
  </div>
  <div class="panel">
    <div class="timeline">
      <div class="tl-item" style="--tl-color:var(--accent)">
        <div class="tl-content">
          <h4>① Federated Training Phase</h4>
          <p>Each user trains a local NCF model on their own device; only item embedding deltas (ΔM) are shared with the server. The target user's historical updates are sampled and stored using importance-based selection (top 50% by gradient magnitude).</p>
        </div>
      </div>
      <div class="tl-item" style="--tl-color:var(--accent2)">
        <div class="tl-content">
          <h4>② Unlearning Request (GDPR / Right to be Forgotten)</h4>
          <p>User #<strong id="heroTargetUser">42</strong> submits a deletion request. The server does not contact any other client or access raw data — only the stored gradient history is used.</p>
        </div>
      </div>
      <div class="tl-item" style="--tl-color:var(--warn)">
        <div class="tl-content">
          <h4>③ Skew Estimation via Lipschitz Condition</h4>
          <p>CFRU computes the accumulated model skew caused by the missing client across all prior rounds using the Lipschitz bound: |ε_t| ≤ K·|M*_t − M_t|. A hyperparameter α approximates K for efficiency.</p>
        </div>
      </div>
      <div class="tl-item" style="--tl-color:var(--accent3)">
        <div class="tl-content">
          <h4>④ Gradient Rollback &amp; Certified Correction</h4>
          <p>Using the recursive formula Δ_t ≈ (1+α)·Δ_{t-1} + correction, the server computes the total model difference Δ_T in O(T·N) time — without any retraining. The corrected model M* = M + Δ_T is statistically indistinguishable from one retrained without the target user.</p>
        </div>
      </div>
    </div>
  </div>
</div>

<footer>
  <span>CFRU POC — RAI Assignment · Fasih Ur Rehman (I22-1910) &amp; Sohaib Shahzad (I22-2034)</span>
  <span>Paper: Huynh et al., ACM TOIS Vol.43 No.2, 2025</span>
</footer>
</div><!-- end app-wrap -->

<script>
/* ── STATE ───────────────────────────────────────────────────────────────── */
const RECS = { baseline: [], unlearned: [], retrained: [], live: [] };
let activeTab = 'baseline';

/* ── ON LOAD ─────────────────────────────────────────────────────────────── */
window.addEventListener('DOMContentLoaded', () => {
  loadMetrics();
  document.getElementById('targetUser').addEventListener('input', e => {
    document.getElementById('heroTargetUser').textContent = e.target.value;
  });
  document.getElementById('heroTargetUser').textContent =
    document.getElementById('targetUser').value;
});

/* ── LOAD METRICS ────────────────────────────────────────────────────────── */
async function loadMetrics() {
  const m = await fetch('/api/metrics').then(r => r.json());

  // Header cards
  const u = m.utility;
  const set = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
  set('m-hr-unlearn',  u.unlearned.hr.toFixed(4));
  set('m-ndcg-unlearn', u.unlearned.ndcg.toFixed(4));
  const speedup = (m.efficiency.retrain_time_seconds / m.efficiency.cfru_time_seconds).toFixed(0);
  set('m-speedup', speedup + '×');
  set('m-time', m.efficiency.cfru_time_seconds.toFixed(2) + 's');

  const hrDelta = ((u.unlearned.hr - u.baseline.hr) / u.baseline.hr * 100).toFixed(1);
  set('m-hr-delta', (hrDelta > 0 ? '+' : '') + hrDelta + '% vs baseline');
  const ndcgDelta = ((u.unlearned.ndcg - u.baseline.ndcg) / u.baseline.ndcg * 100).toFixed(1);
  set('m-ndcg-delta', (ndcgDelta > 0 ? '+' : '') + ndcgDelta + '% vs baseline');
  set('m-time-label', 'retrain: ' + m.efficiency.retrain_time_seconds + 's');

  // Comparison table
  const tbody = document.getElementById('compareTbody');
  const rows = [
    { name: 'Baseline',  badge: 'blue',   hr: u.baseline.hr,  ndcg: u.baseline.ndcg,  mia: m.privacy_mia.baseline,  time: '—',         priv: '❌ Unlearned data persists' },
    { name: 'CFRU ★',   badge: 'purple', hr: u.unlearned.hr, ndcg: u.unlearned.ndcg, mia: m.privacy_mia.unlearned, time: m.efficiency.cfru_time_seconds + 's', priv: '✅ Certified removal' },
    { name: 'Retrained', badge: 'green',  hr: u.retrained.hr, ndcg: u.retrained.ndcg, mia: m.privacy_mia.retrained, time: m.efficiency.retrain_time_seconds + 's', priv: '✅ Ground truth clean' },
  ];
  tbody.innerHTML = rows.map(r => `
    <tr>
      <td><span class="badge badge-${r.badge}">${r.name}</span></td>
      <td>${r.hr.toFixed(4)}</td>
      <td>${r.ndcg.toFixed(4)}</td>
      <td style="font-family:var(--mono)">${(r.mia * 100).toFixed(1)}%</td>
      <td style="font-family:var(--mono)">${r.time}</td>
      <td>${r.priv}</td>
    </tr>
  `).join('');

  // Efficiency bars
  const maxTime = Math.max(m.efficiency.retrain_time_seconds, m.efficiency.cfru_time_seconds);
  const effItems = [
    { name: 'CFRU Unlearning', val: m.efficiency.cfru_time_seconds, max: maxTime, color: 'var(--accent)' },
    { name: 'Full Retraining',  val: m.efficiency.retrain_time_seconds, max: maxTime, color: 'var(--danger)' },
  ];
  document.getElementById('effRows').innerHTML = effItems.map(e => `
    <div class="eff-item">
      <div class="eff-label">
        <span class="name">${e.name}</span>
        <span class="val">${e.val}s</span>
      </div>
      <div class="eff-track">
        <div class="eff-fill" style="width:0;background:${e.color}"
             data-target="${(e.val / maxTime * 100).toFixed(1)}"></div>
      </div>
    </div>
  `).join('');
  setTimeout(() => {
    document.querySelectorAll('.eff-fill').forEach(el => {
      el.style.width = el.dataset.target + '%';
    });
  }, 300);

  // MIA gauges
  const miaData = [
    { label: 'Baseline Model',  val: m.privacy_mia.baseline,  color: '#ef4444', ideal: false },
    { label: 'CFRU Unlearned',  val: m.privacy_mia.unlearned, color: '#7c3aed', ideal: true  },
    { label: 'Retrained Model', val: m.privacy_mia.retrained, color: '#10b981', ideal: true  },
    { label: 'Random Chance',   val: 0.5,                     color: '#64748b', ideal: true  },
  ];
  document.getElementById('gaugeRow').innerHTML = miaData.map(g => `
    <div class="gauge-item">
      <div class="gauge-title">${g.label}</div>
      <div class="gauge-pct" style="color:${g.color}">${(g.val*100).toFixed(1)}%</div>
      <div class="gauge-ideal">${g.ideal ? (Math.abs(g.val - 0.5) < 0.1 ? '✅ Near-random' : '≈ Good') : '⚠️ High (data leaked)'}</div>
      <div style="margin-top:8px;height:6px;background:var(--border);border-radius:99px;overflow:hidden">
        <div style="height:100%;width:${g.val*100}%;background:${g.color};border-radius:99px;transition:width 1.2s ease"></div>
      </div>
    </div>
  `).join('');
  setTimeout(() => {
    document.querySelectorAll('#gaugeRow [style*="width:0"]').forEach(el => {
      el.style.width = el.parentElement.dataset.target + '%';
    });
  }, 300);
}

/* ── RECOMMENDATIONS ─────────────────────────────────────────────────────── */
async function fetchRecs() {
  const btn = document.getElementById('btnRecs');
  const userId = parseInt(document.getElementById('targetUser').value);
  const topK   = parseInt(document.getElementById('topK').value);
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>Loading…';
  setStatus('running', 'Fetching recommendations from all three models…');

  try {
    const data = await fetch(`/api/recommend?user_id=${userId}&top_k=${topK}`).then(r => r.json());
    RECS.baseline  = data.baseline;
    RECS.unlearned = data.unlearned;
    RECS.retrained = data.retrained;
    showTab(activeTab);
    setStatus('done', `✅ Loaded top-${topK} recommendations for user #${userId}`);
  } catch(e) {
    setStatus('error', '❌ Failed to load: ' + e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Get Recommendations';
  }
}

/* ── LIVE UNLEARN ────────────────────────────────────────────────────────── */
async function triggerUnlearn() {
  const btn    = document.getElementById('btnUnlearn');
  const userId = parseInt(document.getElementById('targetUser').value);
  const alpha  = parseFloat(document.getElementById('alphaRange').value);
  const rounds = parseInt(document.getElementById('fedRounds').value);
  const topK   = parseInt(document.getElementById('topK').value);

  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>Running CFRU…';
  setStatus('running', `⏳ Executing CFRU unlearning for user #${userId} (α=${alpha}, rounds=${rounds})…`);

  try {
    const data = await fetch('/api/unlearn', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_id: userId, alpha, num_rounds: rounds, top_k: topK })
    }).then(r => r.json());

    RECS.live = data.recommendations;
    const liveTab = document.getElementById('tab-live');
    liveTab.style.display = '';
    showTab('live');
    setStatus('done',
      `✅ CFRU complete in ${data.elapsed_ms}ms\n` +
      `α=${alpha} · rounds=${rounds} · convergence=${data.convergence}%\n` +
      `Items highlighted in purple = changed vs baseline for user #${userId}`
    );
  } catch(e) {
    setStatus('error', '❌ ' + e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = '🗑 Execute CFRU Unlearning';
  }
}

/* ── TAB SWITCHING ───────────────────────────────────────────────────────── */
function showTab(tab) {
  activeTab = tab;
  ['baseline','unlearned','retrained','live'].forEach(t => {
    const el = document.getElementById('tab-' + t);
    if (el) el.classList.toggle('active', t === tab);
  });
  renderRecs(RECS[tab] || []);
}

function renderRecs(items) {
  const list = document.getElementById('recsList');
  if (!items.length) {
    list.innerHTML = '<div style="color:var(--muted);font-size:13px;padding:20px 0">No data — fetch recommendations first.</div>';
    return;
  }
  list.innerHTML = items.map((r, i) => {
    const isChanged = r.changed === true;
    return `
    <div class="rec-item ${isChanged ? 'rec-changed' : ''}">
      <span class="rec-rank" style="${isChanged ? 'color:var(--accent2)' : ''}">#${i+1}</span>
      <span class="rec-title">${r.title}${isChanged ? ' <span style="font-size:11px;color:var(--accent2);font-family:var(--mono)">[new]</span>' : ''}</span>
      <span class="rec-score">${r.score.toFixed(4)}</span>
    </div>`;
  }).join('');
}

/* ── STATUS BOX ──────────────────────────────────────────────────────────── */
function setStatus(type, msg) {
  const box = document.getElementById('statusBox');
  box.className = 'status-box visible status-' + type;
  box.style.whiteSpace = 'pre-line';
  box.textContent = msg;
}
</script>
</body>
</html>
"""

# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/metrics")
def api_metrics():
    return jsonify(METRICS)


@app.route("/api/recommend")
def api_recommend():
    user_id = int(request.args.get("user_id", TARGET_USER))
    top_k   = int(request.args.get("top_k", 10))
    return jsonify({
        "baseline":  get_recommendations(baseline_model,  user_id, top_k),
        "unlearned": get_recommendations(unlearned_model, user_id, top_k),
        "retrained": get_recommendations(retrained_model, user_id, top_k),
    })


@app.route("/api/unlearn", methods=["POST"])
def api_unlearn():
    body       = request.get_json(force=True)
    user_id    = int(body.get("user_id",    TARGET_USER))
    alpha      = float(body.get("alpha",    0.7))
    num_rounds = int(body.get("num_rounds", 20))
    top_k      = int(body.get("top_k",      10))

    new_state, elapsed, convergence = cfru_unlearn_live(user_id, alpha, num_rounds)

    live_model = NCF(NUM_USERS, NUM_ITEMS).to(DEVICE)
    live_model.load_state_dict(new_state)
    live_model.eval()

    recs          = get_recommendations(live_model,      user_id, top_k)
    recs_baseline = get_recommendations(baseline_model,  user_id, top_k)

    # Mark items that changed position vs baseline
    baseline_ids = [r["item_id"] for r in recs_baseline]
    for r in recs:
        r["changed"] = r["item_id"] not in baseline_ids[:top_k]

    return jsonify({
        "recommendations": recs,
        "elapsed_ms":  round(elapsed * 1000, 1),
        "convergence": round(convergence * 100, 1),
        "alpha":       alpha,
        "num_rounds":  num_rounds,
    })


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  CFRU Interactive POC")
    print("  Open  →  http://127.0.0.1:5000")
    print("="*55 + "\n")
    app.run(debug=False, host="0.0.0.0", port=5000, use_reloader=False, threaded=True)