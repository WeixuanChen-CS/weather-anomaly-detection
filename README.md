# Weather Anomaly Detection (R, k-NN, Autoencoder)

## Overview

This project implements an end-to-end anomaly detection pipeline on daily weather data (2023).

Two approaches are applied and compared:

- Density-based method (k-NN distance, k = 15)
- Neural network Autoencoder (bottleneck = 2)

The pipeline includes data cleaning, normalization, feature scaling, anomaly scoring, ranking, and time-series visualization.

---

## Tech Stack

- R
- Keras (Neural Networks)
- k-NN distance-based anomaly detection
- Data preprocessing & feature scaling
- Time-series visualization

---

## Project Structure

weather-anomaly-detection/
│
├── src/
│ └── anomaly_pipeline.R
│
├── outputs/
│ ├── AE_bottleneck_2.png
│ └── Density_k15.png
│
├── README.md
└── .gitignore


---

## Key Highlights

- Built structured data processing pipeline for anomaly detection
- Implemented k-NN distance scoring for density-based outlier detection
- Designed and trained autoencoder for reconstruction-based anomaly scoring
- Compared statistical and neural approaches for anomaly ranking
- Generated anomaly score time-series visualizations

---

## Example Output

The generated plots illustrate anomaly scores over time:

- Autoencoder-based anomaly scores
- Density-based anomaly scores (k = 15)

High peaks correspond to extreme weather conditions such as heavy rainfall or unusual visibility patterns.
