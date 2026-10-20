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

.
├── agent/                       # AI agent components
│   ├── coo_agent.py
│   ├── README.md
│   ├── streamlit_app.py
│   └── tools.py
├── charts/                      # Data analysis charts
├── project/
│   ├── app.py                   # Earlier dashboard version
│   ├── app_3.py                 # Main Streamlit application
│   └── app_new.py               # Alternative dashboard version
├── src/
│   └── week8_demo/              # Week 8 demonstration code
├── test/
│   ├── test_calculations.py     # Calculation tests
│   └── test_validation.py       # Data validation tests
├── findings.md                  # Analysis findings
├── Sample.xlsx                  # Input dataset
├── week7_analysis.py            # Week 7 analysis script
├── week8_function_validation.py  # Week 8 functions and validation
├── pyproject.toml               # Project metadata and dependencies
├── uv.lock                      # Locked dependency versions
├── .gitignore                   # Excludes secrets and local artifacts
└── README.md                    # Project documentation

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
uv run streamlit run project/appfinal.py
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


## Project Development Timeline and Individual Contributions

**BA6003 Group 5**

**Project Title:** Corporate Innovation Investment Performance Prediction and Decision Support System

Our project was developed through several stages, from initial topic selection and system design to dashboard development, statistical analysis, AI integration, and portfolio monitoring. The following timeline summarizes the responsibilities and contributions of each team member throughout the project.

### Week 1 — Topic Selection and Initial Project Proposal

**All team members: Ma Lumiao, Ding Liuyuan, and Yang Ruoyu**

- Participated in group discussions and jointly discussed and finalized the project topic.
- Ma Lumiao provide the initial idea and the original database, and data wragling.
- Yang Ruoyu prepared an initial Chinese-language draft based on the proposed project idea.
- Ma Lumiao and Ding Liuyuan subsequently revised, refined, and substantially added and developed the content, translated it into English, and completed the final English version of the proposal.
- Ma Lumiao and Ding Liuyuan uploaded the finalized submission after reviewing and improving the initial draft.


### Week 2 — Foundational System Development, Data Cleaning, and Analytical Framework

**Ding Liuyuan — Foundational System Development and Analytical Visualization**

- Established the foundational structure of the project system and uploaded the initial implementation to the group's GitHub repository.
- Designed and developed the initial system framework to support subsequent analytical and dashboard development.
- Prepared key analytical charts and visualizations to communicate the project's main findings.
- Developed the initial analytical framework for examining the relationship between corporate innovation investment and firm performance.
- Organized the early project workflow and established the technical foundation for subsequent development.

**Ma Lumiao — Data Cleaning and System Framework Refinement**

- Conducted data cleaning to improve the quality, consistency, and usability of the dataset for subsequent analysis.
- Contributed suggestions and improvements to the initial system structure, helping refine the foundational framework established during this stage.
- Reviewed the initial system design and provided recommendations to improve its organization and support subsequent development.
- Collaborated on refining the project's analytical preparation and data presentation.

**Yang Ruoyu — Weekly Submission**

- Submitted the required project form for the week.
### Week 3 — Testing and Data Validation

**Ma Lumiao and Ding Liuyuan — Testing**

- Worked on the initial testing activities to check the functionality and consistency of the early project components.
- Reviewed the initial implementation and helped identify issues requiring further development.

**Yang Ruoyu — Validation**

- Took responsibility for the data validation component.
- Worked on checking the validity and consistency of the relevant data used by the system.

### Week 4 — Initial Dashboard Development and Visualization Framework

**Primary Contributor: Ma Lumiao**

During Week 4, Ma Lumiao took the lead in developing the initial version of the interactive dashboard, transforming the project's underlying data and analytical framework into a user-facing application. This stage established the foundation for the dashboard's subsequent analytical enhancements and AI integration.

- **Dashboard architecture and layout:** Designed and implemented the initial dashboard structure, organizing the interface into clear sections to present company-level financial information, innovation-related indicators, and analytical results in a coherent format.

- **Data integration and presentation:** Connected the available project data to the dashboard and established the initial approach for presenting relevant company-level financial and innovation indicators in a user-friendly format.

- **Key performance indicator (KPI) presentation:** Developed the initial structure for displaying important financial and innovation-related metrics, making the dataset easier to interpret and providing users with a concise overview of company performance.

- **Initial data visualization:** Built the first set of dashboard visualizations to communicate the distributions and relationships in the underlying data. These visualizations provided an accessible way to explore the relationship between corporate innovation investment and firm performance.

- **Analytical results presentation:** Established the initial framework for displaying analytical findings within the dashboard, connecting the project's underlying data analysis with its intended decision-support functionality.

- **User interface and usability:** Developed the initial interface and organized the presentation of data, indicators, and charts to make the application easier to navigate and understand.

- **Foundation for subsequent functionality:** Structured the initial dashboard so that additional analytical features, more detailed visualizations, interactive controls, and AI-assisted analysis could be incorporated during later development stages.

The initial dashboard developed by Ma Lumiao served as the starting point for the subsequent evolution of the application. In the following stage, Ding Liuyuan and Ma Lumiao continued to enhance the dashboard by improving its interface, expanding its analytical functionality, integrating the AI Investment Analyst, and introducing additional scenario-based analysis and supporting documentation.
### Week 5 — Dashboard Enhancement, AI Integration, and System Finalization

During this stage, Ma Lumiao and Ding Liuyuan worked collaboratively to expand the dashboard's functionality, improve the user experience, integrate AI-assisted analysis, and prepare the application and supporting materials for demonstration.

**Ding Liuyuan — AI Integration and Analytical Functionality**

- Integrated the AI Investment Analyst feature into the dashboard, enabling users to obtain supplementary AI-generated analysis of corporate innovation investment and financial performance.
- Configured the OpenRouter API integration and the associated environment-variable settings.
- Enhanced the dashboard's analytical functionality, including the addition of a Tobin's Q scenario-estimation feature based on R&D intensity and patent count.
- Implemented a historical multivariate ordinary least squares (OLS) model to estimate Tobin's Q under user-defined scenarios.
- Added relevant regression information, including R-squared and p-values, to support interpretation of the model results.
- Improved the visualization of the relationship between R&D intensity and Tobin's Q through a scatter plot and OLS trend line.
- Refined the dashboard's analytical presentation and user interactions, including the display of key performance indicators and the relationship between industry selection and the corresponding analytical outputs.
- Improved API error handling and logging to make the AI feature easier to troubleshoot.
- Refined the application code and supporting configuration to improve the overall completeness and usability of the system.

**Ma Lumiao — Dashboard Refinement and Final Version Development**

- Continued developing the initial dashboard into a more complete and polished application.
- Refined the dashboard layout, visual hierarchy, and presentation of key performance indicators.
- Improved the organization and readability of the charts and analytical outputs.
- Enhanced the overall user experience and presentation of the dashboard.
- Integrated the dashboard improvements into the evolving application and contributed to preparing the final version for demonstration.

**Joint Contributions — Documentation, Repository Management, and Project Presentation**

Ma Lumiao and Ding Liuyuan also worked together on the following tasks:

- Updated and refined the project's README to document its objectives, core features, technology stack, setup instructions, and analytical limitations.
- Improved the organization of the online GitHub repository and its supporting project files.
- Refined the application entry point and supporting code to improve the consistency of the project structure.
- Developed and refined the project workflow diagram to illustrate the system's overall process, from data inputs and processing to statistical analysis, dashboard visualization, and AI-assisted interpretation.
- Reviewed the system's overall functionality and prepared the application for demonstration.

The resulting dashboard combines interactive visualizations, statistical analysis, scenario-based estimation, and AI-assisted interpretation in a unified decision-support interface.

### Final Stage — Portfolio Monitoring System

**Primary contributor: Yang Ruoyu**

- Developed the portfolio monitoring system as an additional component of the overall project.
- Contributed functionality related to portfolio monitoring, extending the project's analytical scope beyond the corporate innovation investment dashboard.
- Worked on the testing and validation-related components assigned during the earlier development stages.

### Project Deliverables

The team's work resulted in the following project components:

- **Foundational system:** The initial project structure and analytical framework.
- **Interactive dashboard:** A dashboard for presenting company-level financial indicators and corporate innovation investment analysis.
- **Statistical analysis:** Visualizations and regression-based analysis of the relationship between R&D intensity, patent activity, and Tobin's Q.
- **Scenario estimation:** A feature that estimates Tobin's Q using user-provided R&D intensity and patent count, based on historical regression relationships.
- **AI Investment Analyst:** An AI-assisted analysis feature integrated through OpenRouter.
- **Portfolio monitoring system:** An additional component developed by Yang Ruoyu.
- **Project documentation and workflow:** An updated README, organized repository files, and a workflow diagram describing the system's operation.


### Model Limitations

The Tobin's Q scenario estimates are based on historical statistical relationships and should not be interpreted as validated forecasts of future company performance. Regression statistics and AI-generated commentary are provided to support exploratory analysis and decision-making, not to establish causation or provide investment advice.

## Disclaimer

This project is developed for academic and exploratory decision-support purposes. It is not investment advice, and model outputs should not be used as the sole basis for investment decisions.
