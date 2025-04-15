import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from sklearn.ensemble import RandomForestRegressor
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

# ===== إعدادات التطبيق =====
st.set_page_config(
    layout="wide",
    page_title="نظام إدارة المهام الذكي",
    page_icon="📊",
    menu_items={
        'Get Help': 'https://github.com/yourusername/task-management/issues',
        'About': "نظام إدارة المهام مع ذكاء الاصطناعي - الإصدار 2.0"
    }
)

# ===== ملفات البيانات =====
TASKS_FILE = "tasks.csv"
USERS_FILE = "users.csv"

# ===== ألوان الحالات =====
STATUS_COLORS = {
    "قيد الانتظار": "#FFCC00",
    "قيد التنفيذ": "#3399FF",
    "مكتملة": "#33CC33",
    "ملغاة": "#FF3333"
}

# ===== تحميل النماذج =====
@st.cache_resource(show_spinner="جاري تحميل النماذج الذكية...")
def load_ai_models():
    try:
        # نموذج تحليل المشاعر العربي
        sentiment_model = AutoModelForSequenceClassification.from_pretrained(
            "CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment"
        )
        sentiment_tokenizer = AutoTokenizer.from_pretrained(
            "CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment"
        )
        
        # نموذج تصنيف المهام
        classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device="cpu"
        )
        
        return {
            "sentiment": {
                "model": sentiment_model,
                "tokenizer": sentiment_tokenizer
            },
            "classifier": classifier
        }
    except Exception as e:
        st.error(f"خطأ في تحميل النماذج: {str(e)}")
        return None

# ===== وظائف البيانات =====
def load_data():
    """تحميل البيانات من ملفات CSV"""
    if os.path.exists(TASKS_FILE):
        tasks = pd.read_csv(TASKS_FILE)
        tasks['Start Date'] = pd.to_datetime(tasks['Start Date'])
        tasks['End Date'] = pd.to_datetime(tasks['End Date'])
    else:
        tasks = pd.DataFrame(columns=[
            "Title", "Description", "Assignee", "Status",
            "Priority", "Start Date", "End Date", "Category"
        ])
    
    if os.path.exists(USERS_FILE):
        users = pd.read_csv(USERS_FILE)
    else:
        users = pd.DataFrame([
            {"Username": "admin", "Password": "admin123", "Role": "مدير", "Skills": "إدارة,تخطيط"},
            {"Username": "فني", "Password": "tech123", "Role": "فني", "Skills": "برمجة,أجهزة"},
        ])
    
    return tasks, users

def save_data(tasks, users):
    """حفظ البيانات في ملفات CSV"""
    tasks.to_csv(TASKS_FILE, index=False)
    users.to_csv(USERS_FILE, index=False)

# ===== الواجهة الرئيسية =====
def main():
    # تحميل النماذج الذكية
    ai_models = load_ai_models()
    
    # تحميل البيانات
    tasks, users = load_data()
    
    # نظام المصادقة
    if 'user' not in st.session_state:
        username = st.sidebar.text_input("اسم المستخدم")
        password = st.sidebar.text_input("كلمة المرور", type="password")
        
        if st.sidebar.button("تسجيل الدخول"):
            user = users[(users['Username'] == username) & (users['Password'] == password)]
            if not user.empty:
                st.session_state.user = user.iloc[0].to_dict()
                st.rerun()
            else:
                st.sidebar.error("بيانات الدخول غير صحيحة")
        return
    
    # القائمة الجانبية
    user = st.session_state.user
    st.sidebar.title(f"مرحباً، {user['Username']}")
    st.sidebar.markdown(f"**الدور:** {user['Role']}")
    
    menu_options = ["لوحة التحكم", "إدارة المهام", "لوحة كانبان"]
    if user['Role'] == "مدير":
        menu_options.append("إدارة المستخدمين")
    selected_menu = st.sidebar.radio("القائمة", menu_options)
    
    # صفحات التطبيق
    if selected_menu == "لوحة التحكم":
        show_dashboard(tasks, ai_models)
    elif selected_menu == "إدارة المهام":
        manage_tasks(tasks, users, ai_models)
    elif selected_menu == "لوحة كانبان":
        show_kanban(tasks)
    elif selected_menu == "إدارة المستخدمين":
        manage_users(users)
    
    # تسجيل الخروج
    if st.sidebar.button("تسجيل الخروج"):
        del st.session_state.user
        st.rerun()

# ... [استمرار بقية الدوال كما في الكود السابق] ...

if __name__ == "__main__":
    main()
