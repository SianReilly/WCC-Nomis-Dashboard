# 🏛️ WCC Census Dashboard

An interactive data dashboard built with Python and Streamlit, exploring **Westminster City Council** ward-level data from the **ONS Census 2021** via the Nomis API / modelled Nomis data if the API fails.

Built as part of a workshop introducing data peeps to Python, Streamlit, and GitHub.

---

## 🚀 Live App

👉 [View the dashboard on Streamlit](https://your-app-name.streamlit.app)

> Replace the link above with your Streamlit URL once deployed.

---

## 📊 What It Shows

| Section | Description |
|---|---|
| 🏠 Overview | Deprivation (IMD) by ward, correlation heatmap, Westminster radar chart |
| 👥 Demographics | Age profiles vs London average, ethnicity by ward, population scatter |
| 🏘️ Housing & Tenure | Owner-occupied vs social rented vs private rented by ward, overcrowding analysis |
| 💼 Economy & Labour | Employment rates, industry mix vs London, education vs employment |
| 📊 Statistical Analysis | Linear regression, Random Forest, and K-Means clustering — all interactive |
| 🐍 How It's Built | Annotated code walkthrough of the full stack |

---

## 🛠️ Built With

- **[Streamlit](https://streamlit.io)** — turns the Python script into a web app
- **[pandas](https://pandas.pydata.org)** — data wrangling
- **[Plotly](https://plotly.com/python)** — interactive charts
- **[scikit-learn](https://scikit-learn.org)** — regression, Random Forest, clustering
- **[Nomis API](https://www.nomisweb.co.uk)** — ONS Census 2021 data

---

## 🗂️ Data Source

Data is drawn from the **ONS Census 2021** via the [Nomis web API](https://www.nomisweb.co.uk/api/v01/help). No API key is required for Census 2021 data.

Key datasets used:
- `TS007` — Age by single year
- `TS021` — Ethnic group
- `TS054` — Tenure of household
- `TS060` — Industry (SIC)
- `TS067` — Highest level of qualification

---

## 💻 Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/SianReilly/WCC-Nomis-Dashboard.git
cd WCC-Nomis-Dashboard
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run nomis_dashboard.py
```

The app will open at `http://localhost:8501`

---

## 📁 Repository Structure

```
WCC-Nomis-Dashboard/
│
├── nomis_dashboard.py    # Main Streamlit app
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## 🔄 Deployment

This app is deployed via **Streamlit Community Cloud** (free).

To deploy your own copy:
1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repo, branch `main`, file `nomis_dashboard.py`
5. Click **Deploy**

---

## 👩‍💻 Author

**Sian Reilly** — Westminster City Council · March 2026
