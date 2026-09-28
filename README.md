# Student Management & Performance Analysis System

A Python-based student management and performance analysis system developed as a **Python for Data Science** project.

The system combines **Streamlit, MySQL, Pandas, NumPy, data visualization, and machine learning** to provide an interactive platform for managing student records and analyzing academic performance.

## 📌 Project Overview

Managing student information manually can make it difficult to organize records and analyze academic performance efficiently.

This project provides an interactive web-based system where users can:

* Add student records
* View student records
* Search for students
* Update student information
* Delete student records
* Calculate average marks
* Analyze student performance
* Visualize academic data
* Predict Pass/Fail performance using Machine Learning
* Store student information in a MySQL database

## 🚀 Beyond Syllabus Topic

### Streamlit

**Streamlit** is used as the Beyond Syllabus technology in this project.

It is a Python framework that allows data science and machine learning applications to be converted into interactive web applications without requiring extensive web-development code.

In this project, Streamlit provides the user interface for interacting with the student management system, database, visualizations, and machine-learning components.

## 🛠️ Technologies Used

| Technology      | Purpose                      |
| --------------- | ---------------------------- |
| Python          | Main programming language    |
| Streamlit       | Interactive web interface    |
| MySQL           | Student data storage         |
| MySQL Connector | Python-MySQL connectivity    |
| Pandas          | Data processing and analysis |
| NumPy           | Numerical operations         |
| Matplotlib      | Data visualization           |
| Seaborn         | Statistical visualization    |
| SciPy           | Statistical analysis         |
| Scikit-learn    | Machine learning             |
| Decision Tree   | Pass/Fail prediction         |

## ✨ Features

### 1. Student Management

The system provides complete CRUD operations:

* **Create** – Add new student records
* **Read** – View stored student records
* **Update** – Modify existing information
* **Delete** – Remove student records

### 2. Performance Analysis

The system calculates and displays:

* Subject-wise marks
* Average marks
* Attendance percentage
* Student performance statistics
* Pass/Fail status

### 3. Data Visualization

The project uses Matplotlib and Seaborn to visualize student performance through graphs and charts.

Examples include:

* Subject-wise performance
* Attendance analysis
* Marks distribution
* Performance comparisons
* Correlation analysis

### 4. Machine Learning

A **Decision Tree Classifier** is used to predict whether a student is likely to **Pass or Fail** based on academic and attendance information.

The model uses features such as:

* Python marks
* DSA marks
* Computer Networks marks
* Web Development marks
* Attendance percentage

The average marks are used to determine the Pass/Fail target.

### 5. MySQL Database

Student records are stored in a MySQL database named:

```text
student_management
```

The application automatically creates the required database and `students` table when the database initialization option is used.

## 🗄️ Database Structure

The `students` table contains information such as:

```text
enrollment_no
name
branch
semester
email
mobile
python_marks
dsa_marks
cn_marks
web_dev_marks
average_marks
attendance_pct
created_at
updated_at
```

## 📂 Project Structure

```text
Student-Management-Performance-Analysis/
│
├── app.py
├── README.md
└── screenshots/
    ├── dashboard.png
    ├── student_records.png
    ├── visualizations.png
    └── ml_prediction.png
```

## ⚙️ Installation

### 1. Install Python

Install Python on your system and make sure the Python launcher is available.

### 2. Install Required Libraries

Open the terminal in the project folder and run:

```bash
py -m pip install streamlit mysql-connector-python numpy pandas scipy matplotlib seaborn scikit-learn
```

### 3. Install MySQL

Make sure MySQL Server is installed and running.

MySQL Workbench can be used to verify the database connection.

### 4. Run the Application

Open the project folder in VS Code and run:

```bash
py -m streamlit run app.py
```

The application will open in your web browser.

## 🔐 MySQL Configuration

The Streamlit sidebar provides fields for the MySQL connection.

Typical local configuration:

```text
Host: localhost
Port: 3306
Username: root
Password: Your MySQL password
Database: student_management
```

The password should not be uploaded to GitHub.

## 📊 Machine Learning Workflow

The machine-learning component follows these basic steps:

```text
Student Data
     ↓
Data Cleaning
     ↓
Feature Selection
     ↓
Pass/Fail Target Creation
     ↓
Train/Test Split
     ↓
Decision Tree Classifier
     ↓
Model Evaluation
     ↓
Pass/Fail Prediction
```

## 🎯 Objectives

* To develop a centralized system for managing student records.
* To analyze student academic performance using Python.
* To visualize student performance using data visualization techniques.
* To integrate a MySQL database with a Python application.
* To implement machine learning for Pass/Fail prediction.
* To understand and demonstrate the use of Streamlit for developing interactive data science applications.

## 🔮 Future Scope

The system can be extended with:

* Student login and authentication
* Faculty/admin dashboards
* More advanced machine-learning models
* Automated email notifications
* Attendance alerts
* Subject-wise performance prediction
* Exporting analytical reports
* Cloud database integration
* Deployment of the Streamlit application online

## 👨‍💻 Project

**Project Title:** Student Management & Performance Analysis System

**Subject:** Python for Data Science

**Beyond Syllabus Topic:** Streamlit

**Database:** MySQL

**Application Type:** Interactive Data Science Web Application

## 📚 References

* Python Documentation
* Streamlit Documentation
* Pandas Documentation
* NumPy Documentation
* Matplotlib Documentation
* Seaborn Documentation
* Scikit-learn Documentation
* MySQL Documentation
