"""
Student Management and Performance Analysis System
---------------------------------------------------
GTU Subject : BE05000231 - Python for Data Science
PBL-3      : Micro Project / Application Development

Run:
    pip install streamlit mysql-connector-python pandas numpy scipy matplotlib seaborn scikit-learn reportlab
    streamlit run app.py

MySQL:
    The application can create the database and students table automatically.
    Enter MySQL credentials in the sidebar or set:
        MYSQL_HOST
        MYSQL_PORT
        MYSQL_USER
        MYSQL_PASSWORD
        MYSQL_DATABASE

Notes:
- Pass is defined as average_marks >= 50.
- The ML target is derived from average_marks. To avoid target leakage,
  average_marks itself is NOT used as a model feature.
"""

import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import mysql.connector
from mysql.connector import Error
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Management & Performance Analysis",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONSTANTS
# ============================================================

DEFAULT_DB = os.getenv("MYSQL_DATABASE", "student_management")
DEFAULT_HOST = os.getenv("MYSQL_HOST", "localhost")
DEFAULT_PORT = int(os.getenv("MYSQL_PORT", "3306"))
DEFAULT_USER = os.getenv("MYSQL_USER", "root")
DEFAULT_PASSWORD = os.getenv("MYSQL_PASSWORD", "")

NUMERIC_COLUMNS = [
    "semester",
    "python_marks",
    "dsa_marks",
    "cn_marks",
    "web_dev_marks",
    "average_marks",
    "attendance_pct",
]

MARK_COLUMNS = [
    "python_marks",
    "dsa_marks",
    "cn_marks",
    "web_dev_marks",
]

TABLE_COLUMNS = [
    "enrollment_no",
    "name",
    "branch",
    "semester",
    "email",
    "mobile",
    "python_marks",
    "dsa_marks",
    "cn_marks",
    "web_dev_marks",
    "average_marks",
    "attendance_pct",
]


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            color: #666;
            font-size: 1rem;
            margin-bottom: 1.5rem;
        }

        .metric-card {
            padding: 0.5rem;
            border-radius: 0.7rem;
        }

        div[data-testid="stMetric"] {
            border: 1px solid rgba(128,128,128,0.20);
            padding: 12px;
            border-radius: 10px;
        }

        .stButton > button {
            border-radius: 8px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATABASE FUNCTIONS
# ============================================================

def get_server_connection(
    host: str,
    port: int,
    user: str,
    password: str,
):
    """Create a connection to the MySQL server without selecting a database."""
    return mysql.connector.connect(
        host=host,
        port=port,
        user=user,
        password=password,
    )


def initialize_database(
    host: str,
    port: int,
    user: str,
    password: str,
    database: str,
) -> Tuple[bool, str]:
    """
    Create the database and students table if they do not already exist.

    Returns:
        (success, message)
    """
    connection = None
    cursor = None

    try:
        connection = get_server_connection(host, port, user, password)
        cursor = connection.cursor()

        # Database name is taken from the controlled UI/env input.
        # Backticks prevent issues with reserved words.
        cursor.execute(
            f"CREATE DATABASE IF NOT EXISTS `{database}` "
            "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
        )

        cursor.execute(f"USE `{database}`")

        create_table_sql = """
        CREATE TABLE IF NOT EXISTS students (
            enrollment_no VARCHAR(30) PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            branch VARCHAR(100) NOT NULL,
            semester INT NOT NULL,
            email VARCHAR(150),
            mobile VARCHAR(20),
            python_marks DECIMAL(5,2) NOT NULL,
            dsa_marks DECIMAL(5,2) NOT NULL,
            cn_marks DECIMAL(5,2) NOT NULL,
            web_dev_marks DECIMAL(5,2) NOT NULL,
            average_marks DECIMAL(5,2) NOT NULL,
            attendance_pct DECIMAL(5,2) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ON UPDATE CURRENT_TIMESTAMP
        )
        """

        cursor.execute(create_table_sql)
        connection.commit()

        return True, f"Database '{database}' and table 'students' are ready."

    except Error as exc:
        return False, f"MySQL initialization error: {exc}"

    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None and connection.is_connected():
            connection.close()


def get_connection(
    host: str,
    port: int,
    user: str,
    password: str,
    database: str,
):
    """Return a connection to the selected MySQL database."""
    return mysql.connector.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
    )


def execute_query(
    query: str,
    params: Optional[tuple],
    db_config: dict,
    fetch: bool = False,
):
    """Execute a parameterized SQL query safely."""
    connection = None
    cursor = None

    try:
        connection = get_connection(**db_config)
        cursor = connection.cursor(dictionary=True)

        cursor.execute(query, params or ())

        if fetch:
            return True, cursor.fetchall()

        connection.commit()
        return True, None

    except Error as exc:
        return False, str(exc)

    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None and connection.is_connected():
            connection.close()


def fetch_students(db_config: dict) -> Tuple[bool, object]:
    """Read all student records."""
    query = """
        SELECT enrollment_no, name, branch, semester, email, mobile,
               python_marks, dsa_marks, cn_marks, web_dev_marks,
               average_marks, attendance_pct
        FROM students
        ORDER BY enrollment_no
    """
    return execute_query(query, None, db_config, fetch=True)


def search_students(db_config: dict, keyword: str):
    """Search by enrollment number, name, branch, email, or mobile."""
    query = """
        SELECT enrollment_no, name, branch, semester, email, mobile,
               python_marks, dsa_marks, cn_marks, web_dev_marks,
               average_marks, attendance_pct
        FROM students
        WHERE enrollment_no LIKE %s
           OR name LIKE %s
           OR branch LIKE %s
           OR email LIKE %s
           OR mobile LIKE %s
        ORDER BY enrollment_no
    """
    pattern = f"%{keyword}%"
    params = (pattern, pattern, pattern, pattern, pattern)
    return execute_query(query, params, db_config, fetch=True)


def fetch_student(db_config: dict, enrollment_no: str):
    """Fetch one student by primary key."""
    query = """
        SELECT enrollment_no, name, branch, semester, email, mobile,
               python_marks, dsa_marks, cn_marks, web_dev_marks,
               average_marks, attendance_pct
        FROM students
        WHERE enrollment_no = %s
    """
    success, rows = execute_query(
        query, (enrollment_no,), db_config, fetch=True
    )

    if success and rows:
        return True, rows[0]
    if success:
        return True, None
    return False, rows


def insert_student(db_config: dict, data: dict):
    """Insert a new student record."""
    query = """
        INSERT INTO students (
            enrollment_no, name, branch, semester, email, mobile,
            python_marks, dsa_marks, cn_marks, web_dev_marks,
            average_marks, attendance_pct
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    params = (
        data["enrollment_no"],
        data["name"],
        data["branch"],
        data["semester"],
        data["email"],
        data["mobile"],
        data["python_marks"],
        data["dsa_marks"],
        data["cn_marks"],
        data["web_dev_marks"],
        data["average_marks"],
        data["attendance_pct"],
    )

    return execute_query(query, params, db_config)


def update_student(db_config: dict, enrollment_no: str, data: dict):
    """Update all editable fields for an existing student."""
    query = """
        UPDATE students
        SET name = %s,
            branch = %s,
            semester = %s,
            email = %s,
            mobile = %s,
            python_marks = %s,
            dsa_marks = %s,
            cn_marks = %s,
            web_dev_marks = %s,
            average_marks = %s,
            attendance_pct = %s
        WHERE enrollment_no = %s
    """

    params = (
        data["name"],
        data["branch"],
        data["semester"],
        data["email"],
        data["mobile"],
        data["python_marks"],
        data["dsa_marks"],
        data["cn_marks"],
        data["web_dev_marks"],
        data["average_marks"],
        data["attendance_pct"],
        enrollment_no,
    )

    return execute_query(query, params, db_config)


def delete_student(db_config: dict, enrollment_no: str):
    """Delete a student by enrollment number."""
    query = "DELETE FROM students WHERE enrollment_no = %s"
    return execute_query(query, (enrollment_no,), db_config)


# ============================================================
# DATA PROCESSING / VALIDATION
# ============================================================

def calculate_average(*marks: float) -> float:
    """Calculate average marks across the four subjects."""
    return round(float(np.mean(marks)), 2)


def validate_marks(value: float, field_name: str) -> Optional[str]:
    """Validate marks are in the 0-100 range."""
    if value < 0 or value > 100:
        return f"{field_name} must be between 0 and 100."
    return None


def validate_attendance(value: float) -> Optional[str]:
    """Validate attendance percentage."""
    if value < 0 or value > 100:
        return "Attendance must be between 0 and 100."
    return None


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Convert numeric fields to numeric dtype and handle missing values."""
    if df.empty:
        return df

    result = df.copy()

    for column in NUMERIC_COLUMNS:
        if column in result.columns:
            result[column] = pd.to_numeric(
                result[column], errors="coerce"
            )

    result = result.dropna(subset=NUMERIC_COLUMNS)

    # Recalculate average from the four subject marks so analysis is
    # consistent with the source subject scores.
    if all(col in result.columns for col in MARK_COLUMNS):
        result["calculated_average"] = result[MARK_COLUMNS].mean(axis=1)

    return result


# ============================================================
# STATISTICS
# ============================================================

def calculate_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean, median, minimum, maximum, standard deviation and mode
    for every numeric column.
    """
    numeric_df = df.select_dtypes(include=np.number)

    if numeric_df.empty:
        return pd.DataFrame()

    rows = []

    for column in numeric_df.columns:
        series = numeric_df[column].dropna()

        if series.empty:
            continue

        mode_result = stats.mode(series.to_numpy(), keepdims=False)

        # scipy.stats.mode returns a scalar for current SciPy versions.
        mode_value = float(np.asarray(mode_result.mode).reshape(-1)[0])

        rows.append(
            {
                "Metric": column,
                "Mean": round(float(np.mean(series)), 2),
                "Median": round(float(np.median(series)), 2),
                "Minimum": round(float(np.min(series)), 2),
                "Maximum": round(float(np.max(series)), 2),
                "Standard Deviation": round(float(np.std(series, ddof=1)), 2)
                if len(series) > 1
                else 0.0,
                "Mode": round(mode_value, 2),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# MACHINE LEARNING
# ============================================================

def train_ml_model(df: pd.DataFrame):
    """
    Train a Decision Tree classifier for Pass/Fail prediction.

    Target:
        1 = Pass, average_marks >= 50
        0 = Fail, average_marks < 50

    average_marks is deliberately excluded from X because the target
    is directly calculated from it. Subject marks and attendance are
    used as predictive features instead.
    """
    required = MARK_COLUMNS + ["attendance_pct", "average_marks"]

    if any(column not in df.columns for column in required):
        return None, "Required ML columns are missing."

    model_df = df[required].dropna().copy()

    if len(model_df) < 6:
        return None, "At least 6 complete student records are recommended for ML."

    model_df["result"] = (
        model_df["average_marks"] >= 50
    ).astype(int)

    if model_df["result"].nunique() < 2:
        return None, (
            "ML requires both Pass and Fail records. "
            "Add students on both sides of the 50-mark threshold."
        )

    X = model_df[MARK_COLUMNS + ["attendance_pct"]]
    y = model_df["result"]

    # Stratification keeps both classes represented in the train/test split
    # when the class counts permit it.
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
            stratify=y,
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
        )

    model = DecisionTreeClassifier(
        random_state=42,
        max_depth=4,
        min_samples_leaf=1,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Fail", "Pass"],
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    )

    return {
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
        "predictions": predictions,
        "accuracy": accuracy,
        "report": report,
        "matrix": matrix,
        "feature_names": list(X.columns),
        "dataset": model_df,
    }, None


# ============================================================
# PDF REPORT GENERATION - BEYOND SYLLABUS FEATURE
# ============================================================

def generate_student_pdf(student: dict) -> bytes:
    """Generate an individual student performance report as a PDF."""
    buffer = __import__("io").BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
        title=f"Student Performance Report - {student['name']}",
        author="Student Management & Performance Analysis System",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=18,
        spaceAfter=8,
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=9,
        textColor=colors.grey,
        spaceAfter=18,
    )
    heading_style = ParagraphStyle(
        "ReportHeading",
        parent=styles["Heading2"],
        fontSize=12,
        spaceBefore=10,
        spaceAfter=8,
    )
    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontSize=9,
        leading=13,
    )

    story = []
    story.append(Paragraph("STUDENT PERFORMANCE REPORT", title_style))
    story.append(
        Paragraph(
            "GTU BE05000231 - Python for Data Science | PBL-3",
            subtitle_style,
        )
    )

    details = [
        ["Student Name", str(student["name"]), "Enrollment No.", str(student["enrollment_no"])],
        ["Branch", str(student["branch"]), "Semester", str(student["semester"])],
        ["Email", str(student.get("email") or "-"), "Mobile", str(student.get("mobile") or "-")],
        ["Attendance", f"{float(student['attendance_pct']):.2f}%", "Average Marks", f"{float(student['average_marks']):.2f}"],
    ]

    details_table = Table(details, colWidths=[1.25*inch, 2.35*inch, 1.25*inch, 2.0*inch])
    details_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
        ("BACKGROUND", (2, 0), (2, -1), colors.whitesmoke),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))

    story.append(Paragraph("Student Details", heading_style))
    story.append(details_table)
    story.append(Spacer(1, 10))

    subject_rows = [
        ["Subject", "Marks"],
        ["Python", f"{float(student['python_marks']):.2f}"],
        ["Data Structures & Algorithms", f"{float(student['dsa_marks']):.2f}"],
        ["Computer Networks", f"{float(student['cn_marks']):.2f}"],
        ["Web Development", f"{float(student['web_dev_marks']):.2f}"],
        ["Average", f"{float(student['average_marks']):.2f}"],
    ]

    subject_table = Table(subject_rows, colWidths=[4.9*inch, 1.95*inch])
    subject_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
        ("ALIGN", (1, 1), (1, -1), "CENTER"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))

    story.append(Paragraph("Subject-wise Performance", heading_style))
    story.append(subject_table)
    story.append(Spacer(1, 10))

    average = float(student["average_marks"])
    attendance = float(student["attendance_pct"])
    result = "PASS" if average >= 50 else "FAIL"

    if average >= 75:
        observation = "The student has demonstrated strong overall academic performance."
    elif average >= 50:
        observation = "The student has demonstrated satisfactory overall academic performance."
    else:
        observation = "The student may benefit from additional academic support and practice."

    summary = [
        ["Result", result],
        ["Average Marks", f"{average:.2f} / 100"],
        ["Attendance", f"{attendance:.2f}%"],
        ["Observation", observation],
    ]

    summary_table = Table(summary, colWidths=[1.55*inch, 5.3*inch])
    summary_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))

    story.append(Paragraph("Performance Summary", heading_style))
    story.append(summary_table)
    story.append(Spacer(1, 14))

    # Add a compact subject-marks chart to make the generated report more useful.
    chart_buffer = __import__("io").BytesIO()
    chart_labels = ["Python", "DSA", "CN", "Web Dev"]
    chart_values = [
        float(student["python_marks"]),
        float(student["dsa_marks"]),
        float(student["cn_marks"]),
        float(student["web_dev_marks"]),
    ]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(chart_labels, chart_values)
    ax.set_title("Subject-wise Marks")
    ax.set_ylabel("Marks")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(chart_buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    chart_buffer.seek(0)

    story.append(Paragraph("Performance Chart", heading_style))
    story.append(Image(chart_buffer, width=6.5*inch, height=3.0*inch))
    story.append(Spacer(1, 10))
    story.append(
        Paragraph(
            "This report was generated automatically by the Student Management & "
            "Performance Analysis System using Python and ReportLab.",
            subtitle_style,
        )
    )

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def page_pdf_reports(db_config: dict):
    """Generate and download individual student performance reports."""
    st.header("📄 Student Performance Report")
    st.write(
        "Generate a structured PDF report containing student details, "
        "subject marks, average, attendance, result and a performance chart."
    )

    success, result = fetch_students(db_config)

    if not success:
        st.error(f"Could not load student records: {result}")
        return

    students = result

    if not students:
        st.info("Add student records before generating a report.")
        return

    student_options = {
        f"{row['enrollment_no']} - {row['name']}": row["enrollment_no"]
        for row in students
    }

    selected_label = st.selectbox(
        "Select Student",
        list(student_options.keys()),
    )
    selected_enrollment = student_options[selected_label]

    found, student = fetch_student(db_config, selected_enrollment)

    if not found:
        st.error(f"Could not load student: {student}")
        return

    if student is None:
        st.warning("Selected student could not be found.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Average Marks", f"{float(student['average_marks']):.2f}")
    with c2:
        st.metric("Attendance", f"{float(student['attendance_pct']):.2f}%")
    with c3:
        result_text = "PASS" if float(student["average_marks"]) >= 50 else "FAIL"
        st.metric("Result", result_text)

    st.subheader("Report Preview")
    preview = pd.DataFrame([
        {
            "Student": student["name"],
            "Enrollment No.": student["enrollment_no"],
            "Python": float(student["python_marks"]),
            "DSA": float(student["dsa_marks"]),
            "CN": float(student["cn_marks"]),
            "Web Development": float(student["web_dev_marks"]),
            "Average": float(student["average_marks"]),
            "Attendance %": float(student["attendance_pct"]),
        }
    ])
    st.dataframe(preview.round(2), use_container_width=True, hide_index=True)

    pdf_bytes = generate_student_pdf(student)

    st.download_button(
        label="⬇️ Download PDF Performance Report",
        data=pdf_bytes,
        file_name=f"student_report_{student['enrollment_no']}.pdf",
        mime="application/pdf",
        use_container_width=True,
        type="primary",
    )


# ============================================================
# UI HELPERS
# ============================================================

def render_header():
    """Render the common application header."""
    st.markdown(
        '<div class="main-title">🎓 Student Management & Performance Analysis System</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="subtitle">GTU BE05000231 • Python for Data Science • PBL-3</div>',
        unsafe_allow_html=True,
    )


def number_input_marks(label: str, value: float = 0.0, key: str = ""):
    """Reusable marks input."""
    return st.number_input(
        label,
        min_value=0.0,
        max_value=100.0,
        value=float(value),
        step=0.5,
        key=key,
    )


def student_form(defaults: Optional[dict] = None, disabled_enrollment=False):
    """Render the student input form and return submitted form values."""
    defaults = defaults or {}

    with st.form("student_form", clear_on_submit=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            enrollment_no = st.text_input(
                "Enrollment No. *",
                value=str(defaults.get("enrollment_no", "")),
                disabled=disabled_enrollment,
            )
            name = st.text_input(
                "Student Name *",
                value=str(defaults.get("name", "")),
            )
            branch = st.text_input(
                "Branch *",
                value=str(defaults.get("branch", "Computer Engineering")),
            )
            semester = st.number_input(
                "Semester *",
                min_value=1,
                max_value=8,
                value=int(defaults.get("semester", 5)),
                step=1,
            )

        with col2:
            email = st.text_input(
                "Email",
                value=str(defaults.get("email", "")),
            )
            mobile = st.text_input(
                "Mobile",
                value=str(defaults.get("mobile", "")),
            )
            python_marks = number_input_marks(
                "Python Marks",
                defaults.get("python_marks", 0),
                "python_marks",
            )
            dsa_marks = number_input_marks(
                "DSA Marks",
                defaults.get("dsa_marks", 0),
                "dsa_marks",
            )

        with col3:
            cn_marks = number_input_marks(
                "CN Marks",
                defaults.get("cn_marks", 0),
                "cn_marks",
            )
            web_dev_marks = number_input_marks(
                "Web Development Marks",
                defaults.get("web_dev_marks", 0),
                "web_dev_marks",
            )

            calculated = calculate_average(
                python_marks,
                dsa_marks,
                cn_marks,
                web_dev_marks,
            )

            st.metric("Calculated Average", f"{calculated:.2f}")

            attendance_pct = st.number_input(
                "Attendance (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(defaults.get("attendance_pct", 75.0)),
                step=0.5,
            )

        submitted = st.form_submit_button(
            "Save Student Record",
            use_container_width=True,
            type="primary",
        )

    data = {
        "enrollment_no": enrollment_no.strip(),
        "name": name.strip(),
        "branch": branch.strip(),
        "semester": int(semester),
        "email": email.strip(),
        "mobile": mobile.strip(),
        "python_marks": python_marks,
        "dsa_marks": dsa_marks,
        "cn_marks": cn_marks,
        "web_dev_marks": web_dev_marks,
        "average_marks": calculated,
        "attendance_pct": attendance_pct,
    }

    return submitted, data


def validate_student_data(data: dict) -> list:
    """Return a list of validation errors."""
    errors = []

    if not data["enrollment_no"]:
        errors.append("Enrollment number is required.")

    if not data["name"]:
        errors.append("Student name is required.")

    if not data["branch"]:
        errors.append("Branch is required.")

    for field in MARK_COLUMNS:
        error = validate_marks(data[field], field.replace("_", " ").title())
        if error:
            errors.append(error)

    attendance_error = validate_attendance(data["attendance_pct"])
    if attendance_error:
        errors.append(attendance_error)

    if data["email"] and "@" not in data["email"]:
        errors.append("Please enter a valid email address.")

    return errors


def display_student_table(df: pd.DataFrame):
    """Display a clean dataframe in Streamlit."""
    if df.empty:
        st.info("No student records found.")
        return

    formatted = df.copy()

    for col in [
        "python_marks",
        "dsa_marks",
        "cn_marks",
        "web_dev_marks",
        "average_marks",
        "attendance_pct",
    ]:
        if col in formatted.columns:
            formatted[col] = formatted[col].astype(float).round(2)

    st.dataframe(
        formatted,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# PAGE: MANAGE STUDENT RECORDS
# ============================================================

def page_manage_students(db_config: dict):
    """CRUD page."""
    st.header("Manage Student Records")

    tab_add, tab_view, tab_search, tab_update, tab_delete = st.tabs(
        ["➕ Add", "📋 View", "🔎 Search", "✏️ Update", "🗑️ Delete"]
    )

    # ---------------- ADD ----------------
    with tab_add:
        st.subheader("Add New Student")

        submitted, data = student_form()

        if submitted:
            errors = validate_student_data(data)

            if errors:
                for error in errors:
                    st.error(error)
            else:
                success, result = insert_student(db_config, data)

                if success:
                    st.success(
                        f"Student {data['enrollment_no']} added successfully."
                    )
                else:
                    st.error(f"Could not add student: {result}")

    # ---------------- VIEW ----------------
    with tab_view:
        st.subheader("All Student Records")

        if st.button("🔄 Refresh Records", key="refresh_records"):
            st.rerun()

        success, result = fetch_students(db_config)

        if success:
            df = pd.DataFrame(result)
            display_student_table(df)

            if not df.empty:
                st.caption(f"Total records: {len(df)}")
        else:
            st.error(f"Could not load records: {result}")

    # ---------------- SEARCH ----------------
    with tab_search:
        st.subheader("Search Student")

        keyword = st.text_input(
            "Search by Enrollment No., Name, Branch, Email or Mobile",
            key="search_keyword",
        )

        if keyword.strip():
            success, result = search_students(
                db_config,
                keyword.strip(),
            )

            if success:
                display_student_table(pd.DataFrame(result))
            else:
                st.error(f"Search failed: {result}")
        else:
            st.info("Enter a search value to find student records.")

    # ---------------- UPDATE ----------------
    with tab_update:
        st.subheader("Update Existing Student")

        update_enrollment = st.text_input(
            "Enter Enrollment No. to Update",
            key="update_enrollment",
        )

        if update_enrollment.strip():
            found, student = fetch_student(
                db_config,
                update_enrollment.strip(),
            )

            if not found:
                st.error(f"Could not fetch student: {student}")
            elif student is None:
                st.warning("No student found with that enrollment number.")
            else:
                submitted, data = student_form(
                    student,
                    disabled_enrollment=True,
                )

                if submitted:
                    data["enrollment_no"] = update_enrollment.strip()
                    errors = validate_student_data(data)

                    if errors:
                        for error in errors:
                            st.error(error)
                    else:
                        success, result = update_student(
                            db_config,
                            update_enrollment.strip(),
                            data,
                        )

                        if success:
                            st.success("Student record updated successfully.")
                        else:
                            st.error(f"Update failed: {result}")

    # ---------------- DELETE ----------------
    with tab_delete:
        st.subheader("Delete Student")

        delete_enrollment = st.text_input(
            "Enter Enrollment No. to Delete",
            key="delete_enrollment",
        )

        if delete_enrollment.strip():
            found, student = fetch_student(
                db_config,
                delete_enrollment.strip(),
            )

            if not found:
                st.error(f"Could not fetch student: {student}")
            elif student is None:
                st.warning("No student found with that enrollment number.")
            else:
                st.warning(
                    f"Record selected: {student['name']} "
                    f"({student['enrollment_no']})"
                )

                confirm = st.checkbox(
                    "I confirm that I want to permanently delete this record.",
                    key="delete_confirm",
                )

                if st.button(
                    "Delete Record",
                    type="secondary",
                    disabled=not confirm,
                    key="delete_button",
                ):
                    success, result = delete_student(
                        db_config,
                        delete_enrollment.strip(),
                    )

                    if success:
                        st.success("Student record deleted successfully.")
                    else:
                        st.error(f"Delete failed: {result}")


# ============================================================
# PAGE: DESCRIPTIVE STATISTICS
# ============================================================

def page_statistics(db_config: dict):
    """Statistical analysis page."""
    st.header("Descriptive Statistics")

    success, result = fetch_students(db_config)

    if not success:
        st.error(f"Could not load data: {result}")
        return

    df = clean_dataframe(pd.DataFrame(result))

    if df.empty:
        st.info("Add student records before performing analysis.")
        return

    st.subheader("Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Students", len(df))

    with c2:
        st.metric(
            "Overall Average",
            f"{df['average_marks'].mean():.2f}",
        )

    with c3:
        st.metric(
            "Average Attendance",
            f"{df['attendance_pct'].mean():.2f}%",
        )

    with c4:
        pass_rate = (df["average_marks"] >= 50).mean() * 100
        st.metric("Pass Rate", f"{pass_rate:.2f}%")

    st.subheader("Mean, Median, Minimum, Maximum, Standard Deviation and Mode")

    stats_df = calculate_statistics(df)

    if not stats_df.empty:
        st.dataframe(
            stats_df,
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Pandas Descriptive Summary")
    st.dataframe(
        df[NUMERIC_COLUMNS].describe().T.round(2),
        use_container_width=True,
    )


# ============================================================
# PAGE: DATA VISUALIZATIONS
# ============================================================

def page_visualizations(db_config: dict):
    """Visualization page with three required plots."""
    st.header("Data Visualizations")

    success, result = fetch_students(db_config)

    if not success:
        st.error(f"Could not load data: {result}")
        return

    df = clean_dataframe(pd.DataFrame(result))

    if df.empty:
        st.info("Add student records before creating visualizations.")
        return

    sns.set_theme(style="whitegrid")

    # --------------------------------------------------------
    # 1. Student Average Performance - Bar Chart
    # --------------------------------------------------------
    st.subheader("1. Student Average Performance")

    fig1, ax1 = plt.subplots(figsize=(12, 5))

    plot_df = df.sort_values("average_marks", ascending=False)

    sns.barplot(
        data=plot_df,
        x="enrollment_no",
        y="average_marks",
        ax=ax1,
    )

    ax1.set_title("Student Average Performance")
    ax1.set_xlabel("Enrollment Number")
    ax1.set_ylabel("Average Marks")
    ax1.set_ylim(0, 100)
    ax1.tick_params(axis="x", rotation=45)

    st.pyplot(fig1, use_container_width=True)
    plt.close(fig1)

    # --------------------------------------------------------
    # 2. Subject-wise Average Marks Comparison - Bar Chart
    # --------------------------------------------------------
    st.subheader("2. Subject-wise Average Marks Comparison")

    subject_means = (
        df[MARK_COLUMNS]
        .mean()
        .rename(
            {
                "python_marks": "Python",
                "dsa_marks": "DSA",
                "cn_marks": "CN",
                "web_dev_marks": "Web Development",
            }
        )
        .reset_index()
    )

    subject_means.columns = ["Subject", "Average Marks"]

    fig2, ax2 = plt.subplots(figsize=(9, 5))

    sns.barplot(
        data=subject_means,
        x="Subject",
        y="Average Marks",
        ax=ax2,
    )

    ax2.set_title("Subject-wise Average Marks Comparison")
    ax2.set_xlabel("Subject")
    ax2.set_ylabel("Average Marks")
    ax2.set_ylim(0, 100)

    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

    # --------------------------------------------------------
    # 3. Attendance vs Average Marks - Scatter Plot
    # --------------------------------------------------------
    st.subheader("3. Attendance vs. Average Marks")

    fig3, ax3 = plt.subplots(figsize=(9, 5))

    sns.scatterplot(
        data=df,
        x="attendance_pct",
        y="average_marks",
        s=90,
        ax=ax3,
    )

    ax3.set_title("Attendance vs. Average Marks")
    ax3.set_xlabel("Attendance (%)")
    ax3.set_ylabel("Average Marks")
    ax3.set_xlim(0, 100)
    ax3.set_ylim(0, 100)

    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

    # Pearson correlation is useful as a supplementary analytical value.
    if len(df) >= 2:
        correlation = df["attendance_pct"].corr(
            df["average_marks"],
            method="pearson",
        )

        st.info(
            f"Pearson correlation between attendance and average marks: "
            f"{correlation:.3f}"
        )


# ============================================================
# PAGE: MACHINE LEARNING
# ============================================================

def page_ml(db_config: dict):
    """Decision Tree Pass/Fail prediction page."""
    st.header("ML Prediction Model")

    st.write(
        "Decision Tree Classifier for Pass/Fail prediction. "
        "Pass = average marks ≥ 50."
    )

    success, result = fetch_students(db_config)

    if not success:
        st.error(f"Could not load data: {result}")
        return

    df = clean_dataframe(pd.DataFrame(result))

    if df.empty:
        st.info("Add student records before training the model.")
        return

    ml_result, error = train_ml_model(df)

    if error:
        st.warning(error)
        return

    accuracy = ml_result["accuracy"]

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Model", "Decision Tree")

    with c2:
        st.metric(
            "Test Accuracy",
            f"{accuracy * 100:.2f}%",
        )

    with c3:
        st.metric(
            "Test Records",
            len(ml_result["y_test"]),
        )

    st.subheader("Classification Report")

    report_df = pd.DataFrame(ml_result["report"]).T
    st.dataframe(
        report_df.round(3),
        use_container_width=True,
    )

    st.subheader("Confusion Matrix")

    cm = ml_result["matrix"]

    fig_cm, ax_cm = plt.subplots(figsize=(6, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Fail", "Pass"],
        yticklabels=["Fail", "Pass"],
        ax=ax_cm,
    )

    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    ax_cm.set_title("Pass/Fail Confusion Matrix")

    st.pyplot(fig_cm, use_container_width=False)
    plt.close(fig_cm)

    st.subheader("Feature Importance")

    importance_df = pd.DataFrame(
        {
            "Feature": ml_result["feature_names"],
            "Importance": ml_result["model"].feature_importances_,
        }
    ).sort_values(
        "Importance",
        ascending=False,
    )

    st.dataframe(
        importance_df.round(4),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Predict Pass / Fail for a New Student")

    with st.form("prediction_form"):
        p1, p2, p3 = st.columns(3)

        with p1:
            new_python = st.number_input(
                "Python Marks",
                0.0,
                100.0,
                60.0,
                0.5,
            )
            new_dsa = st.number_input(
                "DSA Marks",
                0.0,
                100.0,
                60.0,
                0.5,
            )

        with p2:
            new_cn = st.number_input(
                "CN Marks",
                0.0,
                100.0,
                60.0,
                0.5,
            )
            new_web = st.number_input(
                "Web Development Marks",
                0.0,
                100.0,
                60.0,
                0.5,
            )

        with p3:
            new_attendance = st.number_input(
                "Attendance (%)",
                0.0,
                100.0,
                80.0,
                0.5,
            )

        predict_button = st.form_submit_button(
            "Predict Result",
            use_container_width=True,
            type="primary",
        )

    if predict_button:
        new_data = pd.DataFrame(
            [
                [
                    new_python,
                    new_dsa,
                    new_cn,
                    new_web,
                    new_attendance,
                ]
            ],
            columns=MARK_COLUMNS + ["attendance_pct"],
        )

        prediction = ml_result["model"].predict(new_data)[0]

        if prediction == 1:
            st.success("Prediction: PASS")
        else:
            st.error("Prediction: FAIL")

        st.caption(
            "The prediction is produced by the trained Decision Tree model; "
            "it is not a replacement for official academic evaluation."
        )


# ============================================================
# SIDEBAR / APPLICATION ENTRY POINT
# ============================================================

def main():
    """Main Streamlit application."""
    render_header()

    with st.sidebar:
        st.title("⚙️ Configuration")

        st.subheader("MySQL Connection")

        host = st.text_input(
            "Host",
            value=DEFAULT_HOST,
        )

        port = st.number_input(
            "Port",
            min_value=1,
            max_value=65535,
            value=DEFAULT_PORT,
            step=1,
        )

        user = st.text_input(
            "Username",
            value=DEFAULT_USER,
        )

        password = st.text_input(
            "Password",
            value=DEFAULT_PASSWORD,
            type="password",
        )

        database = st.text_input(
            "Database",
            value=DEFAULT_DB,
        )

        st.divider()

        if st.button(
            "Initialize / Test MySQL",
            use_container_width=True,
        ):
            if not database.strip():
                st.error("Database name cannot be empty.")
            else:
                ok, message = initialize_database(
                    host.strip(),
                    int(port),
                    user.strip(),
                    password,
                    database.strip(),
                )

                if ok:
                    st.success(message)
                else:
                    st.error(message)

        st.divider()

        st.subheader("Navigation")

        page = st.radio(
            "Select Page",
            [
                "Manage Student Records",
                "Descriptive Statistics",
                "Data Visualizations",
                "ML Prediction Model",
                "PDF Performance Report",
            ],
        )

        st.divider()

        st.caption(
            "BE05000231 • Python for Data Science\n\n"
            "PBL-3 • Student Management & Performance Analysis"
        )

    db_config = {
        "host": host.strip(),
        "port": int(port),
        "user": user.strip(),
        "password": password,
        "database": database.strip(),
    }

    # Try the database connection before rendering database-dependent pages.
    if not db_config["database"]:
        st.error("Enter a MySQL database name in the sidebar.")
        return

    connection_error = None

    try:
        connection = get_connection(**db_config)
        if connection.is_connected():
            connection.close()
    except Error as exc:
        connection_error = str(exc)

    if connection_error:
        st.warning(
            "MySQL is not connected. Use the sidebar to verify your "
            "credentials and initialize the database."
        )
        st.code(
            "Typical local setup:\n"
            "Host: localhost\n"
            "Port: 3306\n"
            "User: root\n"
            "Password: your MySQL password\n"
            "Database: student_management"
        )
        return

    if page == "Manage Student Records":
        page_manage_students(db_config)

    elif page == "Descriptive Statistics":
        page_statistics(db_config)

    elif page == "Data Visualizations":
        page_visualizations(db_config)

    elif page == "ML Prediction Model":
        page_ml(db_config)

    elif page == "PDF Performance Report":
        page_pdf_reports(db_config)


if __name__ == "__main__":
    main()
