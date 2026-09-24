import ast
import operator as op
import streamlit as st

from crewai import Agent, LLM
from crewai.tools import BaseTool


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Nexus AI | Modern Study Tutor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CREWAI CACHE WORKAROUND
# =========================================================

try:
    import crewai.llms.cache as crewai_cache
    crewai_cache.mark_cache_breakpoint = lambda message: message
except Exception:
    pass


# =========================================================
# NEON AI UI CUSTOM STYLING (CSS)
# =========================================================

st.markdown("""
<style>
    /* Import Modern Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main Container & Gradient Background Hints */
    .main {
        background: radial-gradient(circle at 20% 20%, rgba(0, 210, 255, 0.05) 0%, transparent 40%),
                    radial-gradient(circle at 80% 80%, rgba(58, 123, 213, 0.05) 0%, transparent 40%);
    }

    /* Glowing Hero Card */
    .hero-card {
        padding: 30px;
        border-radius: 20px;
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(0, 210, 255, 0.3);
        box-shadow: 0 8px 32px 0 rgba(0, 210, 255, 0.15);
        backdrop-filter: blur(12px);
        margin-bottom: 25px;
    }

    .hero-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: rgba(15, 23, 42, 0.5);
        border-radius: 12px;
        color: #94a3b8;
        border: 1px solid rgba(255, 255, 255, 0.05);
        font-weight: 600;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0, 210, 255, 0.2) 0%, rgba(58, 123, 213, 0.2) 100%) !important;
        color: #00d2ff !important;
        border: 1px solid rgba(0, 210, 255, 0.5) !important;
        box-shadow: 0 0 15px rgba(0, 210, 255, 0.2);
    }

    /* Glowing Section Cards */
    .neon-box {
        padding: 20px;
        border-radius: 16px;
        background: rgba(15, 23, 42, 0.5);
        border: 1px solid rgba(0, 210, 255, 0.2);
        box-shadow: inset 0 0 10px rgba(0, 210, 255, 0.05);
        margin-bottom: 20px;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        background: rgba(0, 210, 255, 0.15);
        color: #00d2ff;
        border: 1px solid rgba(0, 210, 255, 0.3);
    }

    /* Chat Messages styling tweaks */
    .stChatMessage {
        border-radius: 14px;
        padding: 12px 18px;
        margin-bottom: 10px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(15, 23, 42, 0.5);
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(0, 210, 255, 0.3);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(0, 210, 255, 0.6);
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# SESSION STATE INITIALIZATION
# =========================================================

defaults = {
    "student_name": "Student",
    "subject": "Data Structures",
    "custom_subject": "",
    "topic": "Binary Search Trees",
    "level": "Beginner",
    # Chat histories for individual tabs
    "chat_learn": [],
    "chat_practice": [],
    "chat_quiz": [],
    "chat_plan": []
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# SUBJECT SELECTION HELPERS
# =========================================================

SUBJECTS = [
    "Computer Science",
    "Data Structures",
    "Artificial Intelligence",
    "Generative AI",
    "Cybersecurity",
    "Computer Networks",
    "Software Engineering",
    "Database Systems",
    "HTML & Web Development",
    "Programming",
    "Operating Systems",
    "Computer Architecture",
    "➕ Add Your Own Subject"
]


def get_subject():
    if st.session_state.subject == "➕ Add Your Own Subject":
        if st.session_state.custom_subject.strip():
            return st.session_state.custom_subject.strip()
        return "Custom Subject"
    return st.session_state.subject


# =========================================================
# TOOLS (Calculator & Planner)
# =========================================================

OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
    ast.USub: op.neg,
    ast.UAdd: op.pos
}


def calculate(expression):
    def evaluate(node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
        if isinstance(node, ast.BinOp):
            if type(node.op) in OPERATORS:
                left = evaluate(node.left)
                right = evaluate(node.right)
                return OPERATORS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp):
            if type(node.op) in OPERATORS:
                return OPERATORS[type(node.op)](evaluate(node.operand))
        raise ValueError("Invalid mathematical expression.")

    tree = ast.parse(expression, mode="eval")
    return evaluate(tree.body)


class CalculatorTool(BaseTool):
    name: str = "Calculator"
    description: str = "Safely perform numerical evaluation for mathematical expressions."

    def _run(self, expression: str) -> str:
        try:
            result = calculate(expression)
            return f"Result: {result}"
        except Exception as error:
            return f"Calculator error: {error}"


class StudyPlannerTool(BaseTool):
    name: str = "Study Planner"
    description: str = "Generates actionable study schedules and break recommendations."

    def _run(self, request: str) -> str:
        return (
            "Create a structured and realistic study timeline based on the user's requested "
            "topics, target goals, and available time."
        )


calculator = CalculatorTool()
study_planner = StudyPlannerTool()


# =========================================================
# LLM & AGENT INITIALIZATION
# =========================================================

def create_llm():
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        st.error("🔑 GROQ_API_KEY is missing from Streamlit Secrets.")
        st.stop()

    return LLM(
        model="groq/openai/gpt-oss-120b",
        api_key=api_key,
        temperature=0.3,
        reasoning_effort="medium"
    )


def create_agent():
    return Agent(
        role="Personal AI Study Tutor",
        goal="Guide university students in understanding complex topics through adaptive learning, targeted practice, and customizable study plans.",
        backstory=(
            "You are an expert AI tutor with a patient, encouraging, and clear teaching style. "
            "You use modern formatted Markdown, tables, structured lists, and code blocks to make complex concepts simple."
        ),
        llm=create_llm(),
        tools=[calculator, study_planner],
        allow_delegation=False,
        verbose=False
    )


# =========================================================
# AI QUERY ENGINE
# =========================================================

def ask_tutor(user_input: str, section_mode: str, chat_history_key: str):
    tutor = create_agent()
    subject = get_subject()
    topic = st.session_state.topic or "General Fundamentals"
    level = st.session_state.level
    student = st.session_state.student_name or "Student"

    # Context Instructions
    instructions = {
        "Learn": (
            "Teach the concept clearly from ground up. Provide conceptual summaries, "
            "visual ASCII diagrams or formatted code, intuitive analogies, and key takeaways."
        ),
        "Practice": (
            "Help the student practice step-by-step. Provide practice problems, hints, and feedback. "
            "If the student requests problems only, do NOT immediately reveal solutions."
        ),
        "Quiz": (
            "Act as an active quiz master. Ask engaging questions one by one or in small sets, "
            "evaluate student answers accurately, grade them, and offer concise explanations."
        ),
        "Study Plan": (
            "Act as an academic planner. Produce structured study roadmaps, complete with "
            "time estimates, daily targets, and spaced repetition/break strategies."
        )
    }

    # Format historical context from session history
    history = st.session_state[chat_history_key][-6:]
    history_str = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history])

    prompt = f"""
Student Name: {student}
Subject: {subject}
Topic: {topic}
Difficulty Level: {level}
Current Mode: {section_mode}

Mode Guidelines:
{instructions.get(section_mode, "Provide clear explanations.")}

Recent Conversation History:
{history_str}

Current Request from {student}:
{user_input}
    """

    try:
        response = tutor.execute_task(prompt)
        return str(response)
    except Exception as err:
        return f"⚡ **System Alert:** Unable to contact the AI tutor. Details: {err}"


# =========================================================
# UI LAYOUT & SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown("<h2 style='color: #00d2ff;'>⚙️ Tutor Settings</h2>", unsafe_allow_html=True)
    
    st.session_state.student_name = st.text_input("👤 Student Name", value=st.session_state.student_name)
    
    st.session_state.subject = st.selectbox("📚 Select Subject", SUBJECTS, index=SUBJECTS.index(st.session_state.subject) if st.session_state.subject in SUBJECTS else 0)
    
    if st.session_state.subject == "➕ Add Your Own Subject":
        st.session_state.custom_subject = st.text_input("✏️ Custom Subject Name", value=st.session_state.custom_subject)
        
    st.session_state.topic = st.text_input("🎯 Specific Topic", value=st.session_state.topic)
    
    st.session_state.level = st.select_slider(
        "⚡ Experience Level",
        options=["Beginner", "Intermediate", "Advanced"],
        value=st.session_state.level
    )
    
    st.markdown("---")
    if st.button("🗑️ Clear Active Sessions", use_container_width=True):
        st.session_state.chat_learn = []
        st.session_state.chat_practice = []
        st.session_state.chat_quiz = []
        st.session_state.chat_plan = []
        st.rerun()

# Dynamic Header Banner
st.markdown(f"""
<div class="hero-card">
    <div class="hero-title">Nexus AI Study Core ⚡</div>
    <div class="hero-subtitle">
        Welcome <span style="color:#00d2ff; font-weight:600;">{st.session_state.student_name or 'Student'}</span> • 
        Subject: <span style="color:#00d2ff; font-weight:600;">{get_subject()}</span> • 
        Topic: <span style="color:#00d2ff; font-weight:600;">{st.session_state.topic or 'General'}</span>
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# RENDER HELPER FOR TAB SECTIONS
# =========================================================

def render_section(mode_label, history_key, default_prompt):
    st.markdown(f"<div class='neon-box'><span class='badge'>{mode_label} Module</span> Active tracking for target topic: <b>{st.session_state.topic or 'General'}</b></div>", unsafe_allow_html=True)

    # Render Existing History
    for message in st.session_state[history_key]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # First-time auto prompt action helper
    if not st.session_state[history_key]:
        if st.button(f"🚀 Initialize {mode_label} Session", key=f"btn_init_{history_key}"):
            st.session_state[history_key].append({"role": "user", "content": default_prompt})
            with st.spinner("⚡ AI Core Processing..."):
                response = ask_tutor(default_prompt, mode_label, history_key)
                st.session_state[history_key].append({"role": "assistant", "content": response})
            st.rerun()

    # Chat Input Interface
    if user_prompt := st.chat_input(f"Ask your {mode_label} Tutor...", key=f"input_{history_key}"):
        st.session_state[history_key].append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            with st.spinner("⚡ Processing insight..."):
                response = ask_tutor(user_prompt, mode_label, history_key)
                st.markdown(response)
                st.session_state[history_key].append({"role": "assistant", "content": response})


# =========================================================
# NAVIGATION TABS
# =========================================================

tab_learn, tab_practice, tab_quiz, tab_plan = st.tabs([
    "📖 Learn", 
    "🧪 Practice", 
    "⚡ Quiz", 
    "📅 Study Plan"
])

with tab_learn:
    render_section(
        mode_label="Learn",
        history_key="chat_learn",
        default_prompt=f"Please teach me the core concepts of {st.session_state.topic or get_subject()} clearly with examples."
    )

with tab_practice:
    render_section(
        mode_label="Practice",
        history_key="chat_practice",
        default_prompt=f"Give me a practical exercise/problem regarding {st.session_state.topic or get_subject()} to solve."
    )

with tab_quiz:
    render_section(
        mode_label="Quiz",
        history_key="chat_quiz",
        default_prompt=f"Start a quick 3-question quiz on {st.session_state.topic or get_subject()} to test my knowledge."
    )

with tab_plan:
    render_section(
        mode_label="Study Plan",
        history_key="chat_plan",
        default_prompt=f"Create a structured weekly study plan to master {st.session_state.topic or get_subject()} at the {st.session_state.level} level."
    )
    
