# Asset Task - Demo Version with Persistent Storage
# Task Management System using Streamlit (English UI)

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import os

st.set_page_config(layout="wide", initial_sidebar_state="expanded")

# File paths
TASKS_FILE = "tasks.csv"
USERS_FILE = "users.csv"

# Load data from files
if os.path.exists(USERS_FILE):
    st.session_state.user_data = pd.read_csv(USERS_FILE)
else:
    st.session_state.user_data = pd.DataFrame([
        {"Username": "admin", "Password": "admin123", "Role": "Manager"},
        {"Username": "supervisor", "Password": "sup123", "Role": "Supervisor"},
        {"Username": "user1", "Password": "user123", "Role": "Employee"},
    ])

if os.path.exists(TASKS_FILE):
    st.session_state.tasks = pd.read_csv(TASKS_FILE)
else:
    st.session_state.tasks = pd.DataFrame([
        {"Title": "Prepare Monthly Report", "Assignee": "user1", "Status": "To Do", "Start Date": "2025-04-14", "End Date": "2025-04-18", "Attachment": ""},
        {"Title": "Device Maintenance Follow-up", "Assignee": "user1", "Status": "Doing", "Start Date": "2025-04-10", "End Date": "2025-04-20", "Attachment": ""},
        {"Title": "Warehouse Cleaning", "Assignee": "supervisor", "Status": "Done", "Start Date": "2025-04-01", "End Date": "2025-04-05", "Attachment": ""},
    ])

# Save functions
def save_tasks():
    st.session_state.tasks.to_csv(TASKS_FILE, index=False)

def save_users():
    st.session_state.user_data.to_csv(USERS_FILE, index=False)

# Session state for auth
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.session_state.role = ""

USERS = {
    row.Username: {"password": row.Password, "role": row.Role}
    for _, row in st.session_state.user_data.iterrows()
}

STATUS_COLORS = {
    "To Do": "#ffcc00",
    "Doing": "#3399ff",
    "Done": "#33cc33",
    "Backlog": "#cccccc"
}
KANBAN_COLUMNS = list(STATUS_COLORS.keys())

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

st.sidebar.title("📌 Menu")
menu_items = ["Dashboard", "Add Task", "View Tasks", "Kanban Board"]
if st.session_state.role == "Manager":
    menu_items.append("Manage Users")
menu_items.append("Logout")
page = st.sidebar.radio("Select an action", menu_items)

st.title("📋 Asset Task Dashboard")
st.markdown(f"Welcome **{st.session_state.username}** | Role: **{st.session_state.role}**")

if page == "Dashboard":
    with st.expander("📊 Overview"):
        df = st.session_state.tasks
        total_tasks = len(df)
        completed = df[df["Status"] == "Done"].shape[0]
        in_progress = df[df["Status"] == "Doing"].shape[0]
        new_tasks = df[df["Status"] == "To Do"].shape[0]
        backlog = df[df["Status"] == "Backlog"].shape[0]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📌 Total", total_tasks)
        col2.metric("✅ Done", completed)
        col3.metric("🚧 Doing", in_progress)
        col4.metric("📥 Backlog", backlog)

if page == "Kanban Board":
    st.subheader("🧩 Kanban Board View")
    for status in KANBAN_COLUMNS:
        st.markdown(f"### {status}")
        col_tasks = st.session_state.tasks[st.session_state.tasks["Status"] == status]
        for _, row in col_tasks.iterrows():
            with st.container(border=True):
                st.markdown(f"**{row['Title']}**")
                st.caption(f"Assigned to: {row['Assignee']}")
                st.caption(f"📅 {row['Start Date']} ➜ {row['End Date']}")

if page == "View Tasks":
    st.subheader("📑 Current Tasks")
    def color_row(row):
        color = STATUS_COLORS.get(row["Status"], "white")
        return [f"background-color: {color}" for _ in row]
    styled_df = st.session_state.tasks.drop(columns=["Attachment"]).style.apply(color_row, axis=1)
    st.dataframe(styled_df, use_container_width=True)

if page == "Add Task":
    st.subheader("➕ Add New Task")
    title = st.text_input("Title")
    assigned_to = st.selectbox("Assignee", options=st.session_state.user_data["Username"].tolist())
    status = st.selectbox("Status", options=KANBAN_COLUMNS)
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
        save_tasks()
        st.success("Task added successfully")

        if uploaded_file:
            os.makedirs("uploads", exist_ok=True)
            with open(f"uploads/{uploaded_file.name}", "wb") as f:
                f.write(uploaded_file.getbuffer())

# Logout logic
if page == "Logout":
    st.session_state.authenticated = False
    st.experimental_rerun()

if page == "Manage Users" and st.session_state.role == "Manager":
    st.subheader("👥 Manage Users")
    st.dataframe(st.session_state.user_data, use_container_width=True)

    with st.form("Add User"):
        new_username = st.text_input("New Username")
        new_password = st.text_input("New Password")
        new_role = st.selectbox("Role", ["Manager", "Supervisor", "Employee"])
        submitted = st.form_submit_button("Add User")
        if submitted:
            if new_username in st.session_state.user_data["Username"].values:
                st.warning("Username already exists")
            else:
                st.session_state.user_data = pd.concat([
                    st.session_state.user_data,
                    pd.DataFrame([{"Username": new_username, "Password": new_password, "Role": new_role}])
                ], ignore_index=True)
                save_users()
                st.success("User added successfully")

    with st.form("Delete User"):
        del_username = st.selectbox("Select user to delete", options=st.session_state.user_data["Username"].tolist())
        delete_submitted = st.form_submit_button("Delete User")
        if delete_submitted:
            st.session_state.user_data = st.session_state.user_data[st.session_state.user_data["Username"] != del_username]
            save_users()
            st.success(f"User '{del_username}' deleted successfully")

