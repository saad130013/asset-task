import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from sklearn.ensemble import RandomForestRegressor
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

# ========== إعدادات التطبيق ==========
st.set_page_config(
    layout="wide",
    page_title="نظام إدارة المهام الذكي",
    page_icon="📊",
    menu_items={
        'Get Help': 'https://github.com/yourusername/task-management/issues',
        'Report a bug': "https://github.com/yourusername/task-management/issues",
        'About': "نظام إدارة المهام مع ذكاء اصطناعي - الإصدار 2.0"
    }
)

# ========== ملفات البيانات ==========
TASKS_FILE = "tasks.csv"
USERS_FILE = "users.csv"

# ========== ألوان الحالات ==========
STATUS_COLORS = {
    "قيد الانتظار": "#FFCC00",
    "قيد التنفيذ": "#3399FF",
    "مكتملة": "#33CC33",
    "ملغاة": "#FF3333"
}

# ========== تحميل النماذج (نسخة مخففة) ==========
@st.cache_resource(show_spinner="جاري تحميل النماذج...")
def load_ai_models():
    try:
        # نموذج تحليل المشاعر (أخف وزناً)
        sentiment_model_name = "CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment"
        sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
        sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
        
        # نموذج التصنيف (يعمل بدون اتصال بالإنترنت)
        classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device="cpu"
        )
        
        return {
            "sentiment": {
                "tokenizer": sentiment_tokenizer,
                "model": sentiment_model
            },
            "classifier": classifier
        }
    except Exception as e:
        st.error(f"خطأ في تحميل النماذج: {str(e)}")
        return None

# ========== وظائف مساعدة ==========
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

# ========== الواجهة الرئيسية ==========
def main():
    # تحميل النماذج
    ai_models = load_ai_models()
    
    # تحميل البيانات
    tasks, users = load_data()
    
    # نظام المصادقة
    if 'user' not in st.session_state:
        show_login_page(users)
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

# ========== صفحات التطبيق ==========
def show_login_page(users):
    """صفحة تسجيل الدخول"""
    st.title("🔐 تسجيل الدخول")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        with st.form("login_form"):
            username = st.text_input("اسم المستخدم")
            password = st.text_input("كلمة المرور", type="password")
            
            if st.form_submit_button("دخول"):
                user = users[(users['Username'] == username) & (users['Password'] == password)]
                if not user.empty:
                    st.session_state.user = user.iloc[0].to_dict()
                    st.rerun()
                else:
                    st.error("بيانات الدخول غير صحيحة")

def show_dashboard(tasks, ai_models):
    """لوحة التحكم الرئيسية"""
    st.title("📊 لوحة التحكم")
    
    # الإحصائيات
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("المهام الكلية", len(tasks))
    col2.metric("المكتملة", len(tasks[tasks['Status'] == "مكتملة"]))
    col3.metric("قيد التنفيذ", len(tasks[tasks['Status'] == "قيد التنفيذ"]))
    col4.metric("المتأخرة", len(tasks[tasks['End Date'] < datetime.now()]))
    
    # تحليل المشاعر (إذا كان النموذج متاحاً)
    if ai_models and not tasks.empty and 'Description' in tasks.columns:
        with st.expander("تحليل المشاعر للملاحظات الأخيرة"):
            sample_tasks = tasks.tail(3)
            for _, task in sample_tasks.iterrows():
                if pd.notna(task['Description']):
                    try:
                        inputs = ai_models["sentiment"]["tokenizer"](
                            task['Description'], 
                            return_tensors="pt",
                            truncation=True,
                            max_length=512
                        )
                        outputs = ai_models["sentiment"]["model"](**inputs)
                        sentiment = ["سلبي", "محايد", "إيجابي"][outputs.logits.argmax()]
                        st.write(f"📝 **{task['Title']}**: {sentiment}")
                    except:
                        st.write(f"📝 **{task['Title']}**: تعذر التحليل")

def manage_tasks(tasks, users, ai_models):
    """إدارة المهام"""
    st.title("📝 إدارة المهام")
    
    tab1, tab2 = st.tabs(["عرض المهام", "إضافة مهمة"])
    
    with tab1:
        # فلترة المهام
        col1, col2 = st.columns(2)
        with col1:
            status_filter = st.multiselect(
                "الحالة",
                options=list(STATUS_COLORS.keys()),
                default=list(STATUS_COLORS.keys())
            )
        with col2:
            assignee_filter = st.multiselect(
                "المسؤول",
                options=users['Username'].unique()
            )
        
        # تطبيق الفلتر
        filtered_tasks = tasks.copy()
        if status_filter:
            filtered_tasks = filtered_tasks[filtered_tasks['Status'].isin(status_filter)]
        if assignee_filter:
            filtered_tasks = filtered_tasks[filtered_tasks['Assignee'].isin(assignee_filter)]
        
        # عرض المهام
        st.dataframe(
            filtered_tasks,
            use_container_width=True,
            column_config={
                "Start Date": st.column_config.DateColumn("تاريخ البدء"),
                "End Date": st.column_config.DateColumn("تاريخ الانتهاء")
            }
        )
    
    with tab2:
        with st.form("add_task_form", clear_on_submit=True):
            st.subheader("➕ مهمة جديدة")
            
            col1, col2 = st.columns(2)
            with col1:
                title = st.text_input("عنوان المهمة*", placeholder="أدخل عنواناً واضحاً")
                assignee = st.selectbox("المسؤول*", options=users['Username'])
            with col2:
                priority = st.selectbox("الأولوية*", options=["عادية", "عاجلة", "مهمة"])
                status = st.selectbox("الحالة*", options=list(STATUS_COLORS.keys()))
            
            description = st.text_area("الوصف", height=100)
            
            col3, col4 = st.columns(2)
            with col3:
                start_date = st.date_input("تاريخ البدء*", value=datetime.now())
            with col4:
                end_date = st.date_input("تاريخ الانتهاء*", value=datetime.now() + timedelta(days=3))
            
            # التصنيف التلقائي
            category = None
            if ai_models and title:
                try:
                    result = ai_models["classifier"](
                        title,
                        candidate_labels=["فنية", "إدارية", "مالية", "تسويقية", "دعم فني"]
                    )
                    category = result['labels'][0]
                except:
                    pass
            
            if st.form_submit_button("حفظ المهمة"):
                if not title:
                    st.error("العنوان مطلوب!")
                else:
                    new_task = {
                        "Title": title,
                        "Description": description,
                        "Assignee": assignee,
                        "Status": status,
                        "Priority": priority,
                        "Start Date": start_date,
                        "End Date": end_date,
                        "Category": category or "غير محدد"
                    }
                    
                    tasks = pd.concat([tasks, pd.DataFrame([new_task])], ignore_index=True)
                    save_data(tasks, users)
                    st.success("تمت إضافة المهمة بنجاح!")
                    st.rerun()

# ... (استمرار بقية الدوال بنفس الأسلوب)

if __name__ == "__main__":
    main()
