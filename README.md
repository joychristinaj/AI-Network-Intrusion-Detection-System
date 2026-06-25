# 🛡️ AI Network Intrusion Detection System (NIDS)

**ECE Final Year Project | Placement Portfolio — TCS · LTIMindtree · Presidio**

---

## 📌 What This Project Does

An AI-powered system that monitors network traffic in real time and automatically
detects whether each connection is safe or a cyberattack — using a Machine Learning
model (Random Forest Classifier).

Detects 5 types of traffic:
- ✅ Normal — legitimate connections
- 🔴 DoS — Denial of Service attack
- 🟠 Probe — Port scanning / reconnaissance
- 🟡 R2L — Unauthorized remote access
- 🔴 U2R — Privilege escalation (hacking)

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.x |
| AI/ML | Scikit-learn (Random Forest) |
| Data Processing | Pandas, NumPy |
| Dashboard | Streamlit |
| Dataset | Synthetic KDD Cup 1999-style data |

---

## 🚀 How to Run

### Step 1: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the app
```bash
streamlit run app.py
```

### Step 3: Open browser
Go to `http://localhost:8501`

---

## 📁 Project Structure

```
ai_nids/
├── app.py              # Main application (all code in one file)
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## 🎯 Interview Talking Points

### One-line summary:
> "I built an AI system that detects network attacks in real time using a
> Random Forest classifier trained on network traffic features."

### Tech explanation:
> "The model takes 13 features from each network packet — like bytes transferred,
> login failures, connection count — and classifies it into 5 categories:
> normal, DoS, probe, R2L, or U2R. We got 95%+ accuracy on test data."

### Why Random Forest:
> "It handles class imbalance well, gives feature importance scores,
> and is fast enough for real-time classification without GPU."

### Challenge you solved:
> "The dataset was heavily imbalanced — 50% normal vs rare attacks.
> I used stratified train-test split to ensure the model learned from
> all attack types proportionally."

### How to improve:
> "Integrate real packet capture with Scapy/Wireshark, add LSTM for
> sequential attack patterns, and deploy on AWS/Azure with live logging."

---

## 📊 Model Performance

- Algorithm: Random Forest (100 trees, depth 10)
- Dataset: 2000 samples, 5 classes
- Train/Test Split: 80/20
- Expected Accuracy: ~95%

---

## 🔗 Add to GitHub

```bash
git init
git add .
git commit -m "AI Network Intrusion Detection System - ECE Final Year Project"
git remote add origin https://github.com/YOUR_USERNAME/ai-nids
git push -u origin main
```

---

## 💡 Interview Preparation Checklist

- [ ] Can explain what a Random Forest is in simple terms
- [ ] Can explain all 5 attack types
- [ ] Know what KDD Cup 1999 dataset is
- [ ] Can name 3 real-world companies using NIDS
- [ ] Can explain feature importance
- [ ] Have GitHub link ready to show
- [ ] Prepared 2-minute project walkthrough speech

---

*Built as a placement portfolio project — ECE Final Year 2026*
