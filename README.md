<div align="center">

# 🛡️ Enterprise AI B2B Fraud Detection Engine
### *Graph Network Analytics, Velocity Metrics & Explainable AI for Corporate Risk Mitigation*

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-FastAPI%20%7C%20Streamlit-green.svg)](https://fastapi.tiangolo.com/)
[![ML Model](https://img.shields.io/badge/ML-LightGBM%20%7C%20SHAP-orange.svg)](https://lightgbm.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

[Architecture Overview](#-architecture--pipeline) •
[Key Features](#-key-innovations) •
[Quickstart](#-quickstart-guide) •
[API Documentation](#-real-time-api-endpoints)

</div>

---

## 📌 Executive Summary
In corporate finance, legacy rules-based engines fail to detect **organized internal fraud schemes** like split-approval bypasses and phantom vendors. 

This project implements an **end-to-end B2B Financial Fraud Detection Engine** designed to process enterprise spend streams, detect high-risk relational anomalies using **Graph Analytics**, predict fraud probabilities on highly imbalanced data (~1.5% fraud rate), and provide **SHAP-driven Explainable AI summaries** for compliance auditors.

---

## 🚀 Key Innovations

| Feature Layer | Technology | Problem Solved |
| :--- | :--- | :--- |
| **Graph Network Analysis** | `NetworkX` | Detects **Ghost Vendors** (employee bank accounts matching vendor destination hashes). |
| **Velocity & Limit Features** | `Pandas / NumPy` | Flags **Approval Bypass** (purchases made right under manager sign-off limits, e.g., $9,850 on a $10,000 threshold). |
| **Imbalance Handling** | `LightGBM + Focal Weighting` | Resolves standard ML bias on skewed financial datasets (98.5% legitimate / 1.5% fraud). |
| **Explainable AI (XAI)** | `SHAP` | Converts complex model feature weights into plain-English auditor briefings. |
| **Interactive Triage UI** | `Streamlit` | Enables human-in-the-loop auditors to review flags, inspect metrics, and submit decisions. |
| **Low-Latency REST API** | `FastAPI + Uvicorn` | Provides a real-time `/score_transaction` endpoint for microservice integration. |

---

## 🏗️ Architecture & Pipeline

```text
┌─────────────────────────┐     ┌──────────────────────────────┐
│  Raw B2B Spend Stream   │ ──> │ Feature Engineering Engine   │
│  (Transactions/Invoices)│     │  • Graph Collisions          │
└─────────────────────────┘     │  • Velocity Ratios (1h/24h)  │
                                └──────────────┬───────────────┘
                                               │
                                               ▼
┌─────────────────────────┐     ┌──────────────────────────────┐
│  Streamlit Triage Dashboard    │ <── │ LightGBM + SHAP Classifier   │
│  & FastAPI Real-Time API│     │  • Risk Probability Score    │
└─────────────────────────┘     │  • Automated Audit Notes     │
                                └──────────────────────────────┘