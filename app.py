# Asset Task - Demo Version
# Task Management System using Streamlit (English UI)

import streamlit as st
import pandas as pd
import os
from datetime import datetime, timedelta

# Temporary user data
USERS = {
    "admin": {"password": "admin123", "role": "Manager"},
    "supervisor": {"password": "sup123", "role": "Supervisor"},
    "user1": {"password": "user123", "role": "Employee"},
}

# Session state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.session_state.role = ""

# Status colors
STATUS_COLORS = {
    "New": "#ffcc00",
    "In Progress": "#3399ff",
    "Completed": "#33cc33",
    "Overdue": "#ff3300"
}

# Login page
if not st.session_state.authenticated:
    st.title("🔐 Login to Asset Task")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in USERS and USERS[username]["password"] == password:
            st.session_state.authenticated = True
            st.session_state.username = username
            st.session_state.role = USERS[username]["role"]
        else:
            st.error("Invalid login credentials")
    st.stop()

# Sample tasks
if "tasks" not in st.session_state:
    st.session_state.tasks = pd.DataFrame([
        {"Title": "Prepare Monthly Report", "Assignee": "user1", "Status": "New", "Start Date": "2025-04-14", "End Date": "2025-04-18", "Attachment": ""},
        {"Title": "Device Maintenance Follow-up", "Assignee": "user1", "Status": "In Progress", "Start Date": "2025-04-10", "End Date": "2025-04-20", "Attachment": ""},
        {"Title": "Warehouse Cleaning", "Assignee": "supervisor", "Status": "Completed", "Start Date": "2025-04-01", "End Date": "2025-04-05", "Attachment": ""},
    ])

st.title("📋 Asset Task Dashboard")
st.markdown(f"Welcome **{st.session_state.username}** | Role: **{st.session_state.role}**")

# Dashboard stats
with st.expander("📊 Overview"):
    df = st.session_state.tasks
    total_tasks = len(df)
    completed = df[df["Status"] == "Completed"].shape[0]
    in_progress = df[df["Status"] == "In Progress"].shape[0]
    new_tasks = df[df["Status"] == "New"].shape[0]
    overdue_tasks = df[df["Status"] == "Overdue"].shape[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📌 Total", total_tasks)
    col2.metric("✅ Completed", completed)
    col3.metric("🚧 In Progress", in_progress)
    col4.metric("🕓 Overdue", overdue_tasks)

# Display tasks
st.subheader("📑 Current Tasks")

def color_row(row):
    color = STATUS_COLORS.get(row["Status"], "white")
    return [f"background-color: {color}" for _ in row]

styled_df = st.session_state.tasks.drop(columns=["Attachment"]).style.apply(color_row, axis=1)
st.dataframe(styled_df, use_container_width=True)

# Add new task
with st.expander("➕ Add New Task"):
    title = st.text_input("Title")
    assigned_to = st.selectbox("Assignee", options=list(USERS.keys()))
    status = st.selectbox("Status", options=["New", "In Progress", "Completed", "Overdue"])
    start_date = st.date_input("Start Date", value=datetime.today())
    end_date = st.date_input("End Date", value=datetime.today() + timedelta(days=3))
    uploaded_file = st.file_uploader("Attach a file (optional)")

    if st.button("Save Task"):
        attachment_name = uploaded_file.name if uploaded_file else ""
        new_task = {
            "Title": title,
            "Assignee": assigned_to,
            "Status": status,
            "Start Date": str(start_date),
            "End Date": str(end_date),
            "Attachment": attachment_name,
        }
        st.session_state.tasks = pd.concat([st.session_state.tasks, pd.DataFrame([new_task])], ignore_index=True)
        st.success("Task added successfully")

        if uploaded_file:
            os.makedirs("uploads", exist_ok=True)
            with open(f"uploads/{uploaded_file.name}", "wb") as f:
                f.write(uploaded_file.getbuffer())
