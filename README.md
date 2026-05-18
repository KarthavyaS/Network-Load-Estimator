# AI-Based Crowd-Aware Network Load Estimator

An AI-powered web application that predicts network load and congestion levels based on the number of connected devices and time of usage. The project combines Networking concepts with Machine Learning techniques to provide intelligent network load estimation through an interactive dashboard.

---

## Overview

Network congestion is a common issue in crowded environments such as universities, offices, airports, and public Wi-Fi zones. This project uses Machine Learning models to estimate network load and classify congestion levels using historical data.

The application is developed using **Python**, **Streamlit**, and **Scikit-learn** and provides:

- Real-time load prediction
- Congestion classification
- Analytics visualization
- Prediction history tracking

---

## Features

- Predict network load using AI models
- Supports:
  - Linear Regression
  - Random Forest Regressor
- Interactive analytics dashboard
- Prediction history storage
- Clean and responsive Streamlit UI
- Fast and lightweight application

---

## Technologies Used

- Python
- Streamlit
- Pandas
- Scikit-learn
- Matplotlib

---

## Project Structure

```bash
AI-Network-Load-Estimator/
│
├── app.py
├── network_load.csv
├── history.csv
├── requirements.txt
└── README.md
```

---

## Machine Learning Models

### Linear Regression

Used for basic linear prediction between:
- Devices
- Time
- Network Load

### Random Forest Regressor

Used for improved prediction accuracy and handling non-linear traffic patterns.

---

## Dataset

The dataset contains:

- Number of connected devices
- Time of usage
- Corresponding network load

### Example Dataset

| Devices | Time | Load |
|----------|------|------|
| 10 | 9 | 0.45 |
| 50 | 14 | 1.20 |
| 80 | 20 | 1.75 |

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/AI-Network-Load-Estimator.git
```

### 2. Navigate to Project Folder

```bash
cd AI-Network-Load-Estimator
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

---

## Application Modules

### Home Page

Displays:
- Total records
- Average network load
- Maximum load

### Prediction Page

Allows users to:
- Select number of devices
- Select time
- Choose ML model
- Predict network load

### Analytics Page

Provides:
- Devices vs Load graph
- Time vs Load graph

### History Page

Displays previous prediction records stored in CSV format.

---

## Congestion Classification

| Predicted Load | Congestion Level |
|----------------|------------------|
| < 0.8 | Low Congestion |
| 0.8 – 1.6 | Medium Congestion |
| > 1.6 | High Congestion |

---

## Future Enhancements

- Real-time network traffic monitoring
- Cloud deployment
- IoT-based device tracking
- Deep Learning integration
- Database integration
- Authentication system
- Real-time analytics dashboard

---

## Learning Outcomes

Through this project:

- Networking fundamentals were applied practically
- Machine learning concepts were implemented
- Streamlit dashboard development skills were improved
- Data visualization and predictive analytics techniques were explored

---

## Author

Developed as part of an AI & Networking Internship Project.

---

## License

This project is for educational and internship purposes.
