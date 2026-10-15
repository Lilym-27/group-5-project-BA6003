# Corporate Innovation Investment Performance Prediction and Decision Support System

**BA6003 Group 5**

An interactive analytics dashboard for exploring how corporate R&D investment and patent activity relate to firm performance. The application combines company-level financial data, statistical analysis, interactive visualizations, scenario estimates, and an optional AI-assisted investment analysis feature.

## Project Overview

Innovation investment can be important for a company's long-term competitiveness, but its relationship with financial performance is not always straightforward. This project provides an exploratory decision-support tool that allows users to examine historical patterns and compare scenarios using R&D intensity and patent activity.

The dashboard is designed for analysis and educational decision support. Its statistical relationships should not be interpreted as proof of causation or as guaranteed forecasts of future company performance.

## Key Features

- **Interactive dashboard:** Explore the dataset and key financial and innovation indicators through a Streamlit interface.
- **Company and innovation analysis:** Examine R&D intensity, patent activity, and firm-performance measures.
- **Interactive visualizations:** Use Plotly charts to explore distributions and relationships in the data.
- **Tobin's Q analysis:** Calculate Tobin's Q using the project's defined formula:

  \[
  \text{Tobin's Q} = \frac{\text{Market Value} + \text{Total Liabilities}}{\text{Total Assets}}
  \]

- **Scenario estimation:** Enter R&D intensity and patent count to obtain an estimated Tobin's Q from a historical multivariate OLS model, where supported by the available data.
- **AI Investment Analyst (optional):** Generate AI-assisted commentary through OpenRouter when a valid API key and an available model/provider are configured.

## Technology Stack

- **Python** — data processing and statistical analysis
- **Streamlit** — interactive web application
- **Plotly** — interactive charts
- **pandas / NumPy** — data manipulation
- **statsmodels** — ordinary least squares (OLS) regression
- **uv** — Python project and dependency management
- **OpenRouter / OpenAI-compatible Python client** — optional AI-assisted analysis

## Repository Structure

```text
.
├── project/
│   └── app.py                    # Streamlit dashboard
├── Sample.xlsx                   # Input dataset (if provided locally)
├── pyproject.toml                # Project metadata and dependencies
├── uv.lock                       # Locked dependency versions
├── .env                          # Local API configuration; do not commit
├── .gitignore                    # Excludes secrets and local artifacts
└── README.md
```

Other analysis scripts, validation files, charts, and project documentation may also be present in the repository.

## Getting Started

### 1. Prerequisites

Install Python and [uv](https://docs.astral.sh/uv/). Use a Python version compatible with the project's `pyproject.toml`.

### 2. Clone the repository

```bash
git clone https://github.com/Lilym-27/group-5-project-BA6003.git
cd group-5-project-BA6003
```

### 3. Install dependencies

```bash
uv sync
```

### 4. Prepare the dataset

Place `Sample.xlsx` in the repository root, unless your local project configuration specifies another supported location. Ensure the workbook contains the columns expected by the application. The dataset may not be included in the public repository.

### 5. Configure the optional AI feature

Create a `.env` file in the repository root and add your own OpenRouter configuration:

```dotenv
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=openrouter/free
```

Replace the example key with your own key. Never commit `.env`, paste API keys into source code, or share them in screenshots. The AI feature depends on the selected model/provider's availability, quotas, and rate limits; the dashboard's core data analysis may still be used without AI if the application supports that mode.

### 6. Launch the dashboard

From the repository root, run:

```bash
uv run streamlit run project/app.py
```

Streamlit will print a local URL in the terminal. Open that URL in your browser to use the dashboard.

## Methodology and Interpretation

The application uses historical company data to explore relationships among innovation inputs and firm performance. The scenario estimate uses a multivariate OLS specification with R&D intensity and patent count as explanatory variables for Tobin's Q.

Please keep the following limitations in mind:

- **Association is not causation.** A regression relationship does not establish that R&D spending or patents cause a change in firm value.
- **Historical fit is not forecast validation.** An estimated Tobin's Q is a model-based scenario estimate, not a guaranteed prediction of next year's performance.
- **Data quality matters.** Missing values, extreme observations, differences across industries, and changes over time can affect estimates.
- **Model statistics require context.** R-squared and p-values should be considered alongside sample size, model assumptions, and the underlying data.
- **AI output may be inaccurate.** Treat AI-generated commentary as supplementary interpretation and verify important claims against the data.

## Security and Data Notes

- Keep API keys and other credentials in a local `.env` file.
- Do not upload `.env`, private logs, backups, or confidential datasets to a public repository.
- Confirm that you have permission to distribute any company-level data before publishing it.
- If the dataset is not committed to the repository, collaborators must obtain an authorized copy separately and place it in the expected location.

## Team

**BA6003 Group 5**  
- Course project: Corporate Innovation Investment Performance Prediction and Decision Support System.
- Mainly contributed by Ma Lumiao and Ding Liuyuan, who constructed the basic structure, operated the dashboard, wrote the readme instructure, and completed the AI tools.
- Another team member, Yang Ruoyu, was responsible for creating the test dataset, performing data validation, and building a portfolio monitoring system, thereby making significant contributions to the team's project.

## Disclaimer

This project is developed for academic and exploratory decision-support purposes. It is not investment advice, and model outputs should not be used as the sole basis for investment decisions.
