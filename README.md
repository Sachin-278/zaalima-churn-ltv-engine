# Zaalima Churn & LTV Engine

## 1. The Core Business Problem

Churn = a customer cancels their subscription/service. For subscription businesses (telecom, SaaS, streaming), churn is expensive because acquiring a new customer costs far more than retaining an existing one.

This project has two goals:

- Predict who will churn (classification problem — will they leave: yes/no?)
- Predict how valuable each customer is (LTV — Lifetime Value — regression problem: how much revenue will they generate?)

Combining both lets a business answer a smarter question than "who might leave?" — it answers "who might leave that we can't afford to lose?" A low-value customer churning matters less than a high-value one. This is why marketing teams love this: it prevents wasting retention budget (discounts, calls, offers) on customers who wouldn't have generated much revenue anyway.

## 2. The Data Source

The Telco Customer Churn Dataset (IBM/Kaggle) is a well-known public dataset with ~7,000 customer rows and columns like:

- Demographics: gender, senior citizen status, dependents
- Account info: tenure (months as customer), contract type (month-to-month / 1-year / 2-year), payment method
- Services: internet type, streaming services, tech support, security add-ons
- Billing: monthly charges, total charges
- Target variable: Churn (Yes/No)

It's popular for this exact kind of project because it's realistic, moderately messy (some blank TotalCharges values), and has enough categorical variety to make feature engineering meaningful.

## 3. Tech Stack — What Each Piece Does

| Tool | Role | Why it's used here |
| ---- | ---- | ------------------ |
| PostgreSQL | Data warehouse | Stores raw + processed customer data persistently, allows SQL querying, mimics a real production data layer |
| SQLAlchemy | Python ORM | Lets Python code talk to PostgreSQL without writing raw SQL everywhere; defines tables as Python classes |
| Pandas | Data manipulation | Cleaning, transforming, aggregating data in-memory |
| Scikit-Learn | ML library | Preprocessing (encoding, scaling), baseline models (Logistic Regression, Random Forest), evaluation metrics |
| XGBoost | Gradient boosting | Usually outperforms simpler models on structured/tabular data like this — the industry standard for tasks like churn prediction |
| SHAP | Explainability | Turns a "black box" model's predictions into human-readable reasons |
| FastAPI | API framework | Exposes the trained model as a web service for real-time or batch prediction |
| Superset / Metabase | BI / visualization | Builds dashboards on top of PostgreSQL without custom frontend code |
| Docker | Containerization | Packages the entire stack so it runs identically anywhere |

## 4. Week-by-Week Breakdown

### Week 1: Data Ingestion & EDA

**Day 1-2 — Setup PostgreSQL + load data**
- Create a database, define schema (tables for customers, billing, services), and load the CSV into PostgreSQL.
- This mimics a real production data layer instead of working only from a clean CSV.

**Day 3-5 — EDA (Exploratory Data Analysis)**
- Understand the data before modeling.
- Explore churn rate by contract type, tenure, monthly charges, and other segments.
- Use Seaborn/Matplotlib for bar charts, histograms, and box plots.

**Day 6-7 — Cleaning & encoding**
- Handle missing values (e.g. blank `TotalCharges` values for zero-tenure customers).
- Encode categorical variables using one-hot or label/ordinal encoding.
- Produce a baseline report summarizing churn rate, average tenure, and key segments.

### Week 2: Feature Engineering & Modeling

**Day 1-3 — Feature engineering**
- Create stronger signals such as:
  - `avg_monthly_usage_vs_charge`
  - `tenure_bucket`
  - `num_services_subscribed`
  - `charges_per_tenure`
- Good feature engineering often matters more than model choice for tabular data.

**Day 4-6 — Train classification models**
- Train Logistic Regression, Random Forest, and XGBoost.
- Use precision, recall, and F1 instead of accuracy because churn is usually imbalanced.
- Prioritize recall when retention is more important than false positives.

**Day 7 — SHAP values**
- Use SHAP to explain individual predictions.
- Translate churn probability into tangible feature contributions.

### Week 3: LTV Calculation & API

**Day 1-3 — LTV regression models**
- Train a separate regression model for customer lifetime value.
- A simple formula: `LTV = Average Monthly Revenue × Expected Customer Lifespan`
- Combine churn probability and LTV to create a risk-value matrix.

**Day 4-7 — FastAPI service**
- Build API endpoints such as:
  - `POST /predict/customer`
  - `POST /predict/batch`
- Return churn probability, predicted LTV, and SHAP explanations.

### Week 4: Visualization & Deployment

**Day 1-3 — Connect BI tool**
- Connect Superset or Metabase directly to PostgreSQL.
- Build dashboards with SQL queries and visualizations.

**Day 4-5 — Dashboards**
- Show overall churn rate, churn risk distribution, LTV distribution, and risk-vs-value segments.
- Display feature importance and model explanations.

**Day 6-7 — Docker + documentation**
- Create a `Dockerfile` and `docker-compose.yml` for the API + PostgreSQL stack.
- Make the project runnable with `docker-compose up`.

## 5. Why This Project Structure Makes Sense

This timeline follows the natural ML project lifecycle: ingest → understand → engineer → model → explain → operationalize → visualize → deploy. Each week builds on the last, and the explainability step (SHAP) plus the business framing (LTV, not just churn) elevate this from a toy Kaggle notebook to something closer to real production data science work.

