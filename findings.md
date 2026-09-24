# Week 7 Data Analysis Findings

## 1. Dataset Overview

The dataset contains 8,972 observations and 12 variables, covering company-level observations from 2010 to 2022.

The dataset includes company information, financial variables, R&D expenditure, and patent grants.

## 2. Patent Distribution

The distribution of patent grants is highly right-skewed. Most observations have relatively low numbers of patents granted, while a small number of observations have very high patent counts.

The maximum number of patents granted is 1,389.

![Patent Distribution](charts/patent_distribution.png)

## 3. R&D Intensity Distribution

R&D intensity is also highly right-skewed. Most observations have relatively low R&D intensity, while a small number of observations have extremely high values.

The median R&D intensity is approximately 5.17%, while the maximum is 41,693.79%.

![R&D Intensity Distribution](charts/rd_intensity_distribution.png)

## 4. R&D Intensity and ROA

The scatter plot shows a weak negative association between R&D intensity and ROA in the filtered sample.

The correlation coefficient is approximately -0.109. This indicates that the linear relationship between the two variables is weak.

![R&D Intensity vs ROA](charts/rd_intensity_vs_roa.png)

## 5. R&D Intensity Over Time

Median R&D intensity generally increased from 2016 to 2022.

The most substantial increases occurred between 2016 and 2019. R&D intensity was relatively stable from 2019 to 2021 before increasing again in 2022.

![R&D Intensity Over Time](charts/rd_intensity_over_time.png)

## 6. R&D Intensity by Industry

Median R&D intensity varies across industries.

Some industries have substantially higher median R&D intensity than others, suggesting that innovation investment patterns differ across industry groups.

![R&D Intensity by Industry](charts/rd_intensity_by_industry.png)

## 7. Potential Data Issues

### Missing R&D expenditure data

R&D expenditure data are unavailable for all observations from 2010 to 2015. Therefore, R&D intensity cannot be calculated for these years.

### Extreme R&D intensity values

R&D intensity contains several extreme observations. There are 67 observations with R&D intensity above 50%, including 38 observations above 100%.

The extreme values are mainly associated with observations where revenue is very low relative to R&D expenditure. These observations should be considered carefully in further analysis and modelling.