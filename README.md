#  Elite Athlete Monitoring & Injury Risk AI

**Live Interactive Dashboard:** [Link will go here]

## Overview
Traditional load management in sports science often relies on static, linear assumptions and frequently ignores the profound physiological shifts of the female athlete triad. 

This project is a production-grade, serverless Machine Learning application engineered to solve that gap. It evaluates Acute:Chronic Workload Ratios (ACWR), Heart Rate Variability (rMSSD), sleep quality, and Menstrual Cycle phases to output real-time, non-linear injury risk probabilities for elite combative and endurance athletes.

## Key Features
* **Dual-Engine Machine Learning:** Utilizes separate Gradient Boosting Classifiers (`scikit-learn`) for male and female athletes to ensure biological accuracy.
* **Female Athlete Triad Integration:** The female algorithm dynamically adjusts risk thresholds based on the metabolic and neuromuscular realities of the Follicular, Ovulatory, Luteal, and Menstrual phases.
* **Explainable AI (XAI):** Extracts exact feature importances, allowing coaching staff to see *why* an athlete is at risk.

## Tech Stack
* **Language:** Python 3
* **Machine Learning:** Scikit-Learn (Gradient Boosting, Cross-Validation)
* **Data Processing:** Pandas, NumPy
* **Frontend UI/UX:** Streamlit
