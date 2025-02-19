# baby-driver

Team Members:

- Nimrat Kaur
- Ona Dubey
- Ronit Kadakia
- Tanmay Nanda
- Tanushi Khandelwal

### Objectives and Significance

The objective of this project is to analyse and forecast the demand for taxi services in Manhattan, with a specific focus on the optimisation of cab distribution using dynamic pricing strategies. By studying historical demand patterns and implementing time-series forecasting, we aim to assist the NYC Taxi Commission in strategically positioning the taxi fleet to maximise revenue.

A critical aspect of this analysis is to explore the relationship between demand fluctuations and surge pricing, identify peak hours and locations, and develop recommendations for fleet management. By leveraging predictive models, particularly the Prophet library for time-series forecasting, we can accurately predict future demand, allowing for the proactive deployment of taxis during high-demand periods.

This project holds significant value for improving taxi fleet operations and optimising the user experience. If successful, the model can be used to reduce wait times for passengers, increase the number of successful rides, and enhance revenue generation for the NYC Taxi Commission, especially during periods of high demand.

### Dataset Description

NYC Taxi and Limousine Commission (TLC), Manhattan wanted to optimise the distribution of the taxi fleet and maximise the revenue. TLC Trip Record Data was made available at request by the NYC government [here](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page). We are using the yellow taxi trip records include fields capturing pick-up and drop-off dates/times, pick-up and drop-off locations, trip distances, itemised fares, rate types, payment types, driver-reported passenger counts and other details on surcharge and taxes (total 19 columns). There are 3,152,677 data entries, including 93,174 null values in 5 columns. The dataset is from 2008 till 2022.

### Proposed Methodology

Data Preprocessing and Exploratory Analysis

- Implement rigorous data cleaning protocols, focusing on critical temporal and spatial variables: pickup date-time, location identifiers, and order quantity metrics.
- Systematically evaluate data integrity through missing value analysis and outlier detection, ensuring temporal consistency and spatial accuracy for subsequent modeling phases.

Time-Series Forecasting Implementation

- Deploy the Prophet forecasting framework for its robust handling of temporal patterns and seasonal variations, inherent in urban transportation dynamics.
- Develop hourly demand prediction models, incorporating multiple variables: temporal seasonality, holiday effects, and location-specific demand patterns.
- Implement comprehensive model validation using industry-standard metrics (RMSE, MAE) to ensure prediction accuracy and reliability.

Dynamic Pricing Analysis and Fleet Optimisation

- Investigate correlations between demand fluctuations and dynamic pricing mechanisms, with particular emphasis on surge pricing impacts during peak periods/rush hours.
- Develop data-driven fleet allocation recommendations based on temporal and spatial demand patterns.
- Integrate predictive insights to optimise resource distribution across Manhattan's diverse micro-markets.

Comparative Model Analysis

- Systematic evaluation of forecasting methodologies through parallel implementation of Prophet and ARIMA models.
- Rigorous validation of model predictions against empirical data to assess forecasting accuracy and model robustness.
- Quantitative assessment of model performance across various temporal scales and geographic segments.
