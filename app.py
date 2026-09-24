import streamlit as st
import ast
import operator

from crewai import Agent, Task, Crew, LLM
from crewai.tools import BaseTool


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Study Tutor Agent",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #666666;
        margin-bottom: 25px;
    }

    .welcome-card {
        padding: 35px;
        border-radius: 20px;
        background: linear-gradient(135deg, #f7f9ff, #eef3ff);
        border: 1px solid #dce4ff;
        margin-bottom: 25px;
    }

    .feature-card {
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #e5e7eb;
        background: white;
        min-height: 150px;
    }

    .mode-card {
        padding: 15px;
        border-radius: 14px;
        background: #f8f9fa;
        border: 1px solid #e5e7eb;
        margin-bottom: 10px;
    }

    .profile-box {
        padding: 15px;
        border-radius: 12px;
        background: #f5f7ff;
        margin-bottom: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# CREWAI / LITELLM COMPATIBILITY FIX
# =========================================================

try:
    import crewai.llms.cache as crewai_cache

    crewai_cache.mark_cache_breakpoint = lambda message: message

except Exception:
    pass


# =========================================================
# SESSION STATE
# =========================================================

if "setup_complete" not in st.session_state:
    st.session_state.setup_complete = False

if "student_name" not in st.session_state:
    st.session_state.student_name = ""

if "subject_option" not in st.session_state:
    st.session_state.subject_option = "Computer Science"

if "custom_subject" not in st.session_state:
    st.session_state.custom_subject = ""

if "current_topic" not in st.session_state:
    st.session_state.current_topic = ""

if "learning_level" not in st.session_state:
    st.session_state.learning_level = "Beginner"

if "mode" not in st.session_state:
    st.session_state.mode = "Learn"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "previous_subject" not in st.session_state:
    st.session_state.previous_subject = ""

if "previous_topic" not in st.session_state:
    st.session_state.previous_topic = ""


# =========================================================
# SUBJECTS
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
    "➕ Add Your Own Subject",
]


# =========================================================
# SAFE CALCULATOR
# =========================================================

allowed_operators = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def safe_calculate(expression):

    try:
        tree = ast.parse(expression, mode="eval")

        def evaluate(node):

            if isinstance(node, ast.Constant):

                if isinstance(node.value, (int, float)):
                    return node.value

                raise ValueError("Invalid number")

            if isinstance(node, ast.BinOp):

                left = evaluate(node.left)
                right = evaluate(node.right)

                operation = allowed_operators.get(type(node.op))

                if operation is None:
                    raise ValueError("Operator not allowed")

                return operation(left, right)

            if isinstance(node, ast.UnaryOp):

                value = evaluate(node.operand)

                operation = allowed_operators.get(type(node.op))

                if operation is None:
                    raise ValueError("Operator not allowed")

                return operation(value)

            raise ValueError("Invalid expression")

        return evaluate(tree.body)

    except Exception as error:
        return f"Error: {error}"


# =========================================================
# CALCULATOR TOOL
# =========================================================

class CalculatorTool(BaseTool):

    name: str = "Calculator"

    description: str = (
        "Use this tool for mathematical calculations. "
        "Input should be a simple mathematical expression."
    )

    def _run(self, expression: str) -> str:

        result = safe_calculate(expression)

        return str(result)


calculator = CalculatorTool()


# =========================================================
# STUDY PLANNER TOOL
# =========================================================

class StudyPlannerTool(BaseTool):

    name: str = "Study Planner"

    description: str = (
        "Create a structured study plan based on subject, "
        "topics, number of days, and available study time."
    )

    def _run(
        self,
        subject: str,
        topics: str,
        days: int,
        hours_per_day: float,
    ) -> str:

        topic_list = [
            topic.strip()
            for topic in topics.split(",")
            if topic.strip()
        ]

        if not topic_list:
            return "Please provide at least one topic."

        plan = []

        for day in range(1, days + 1):

            topic = topic_list[(day - 1) % len(topic_list)]

            plan.append(
                f"Day {day}: {topic} - "
                f"{hours_per_day} hour(s)"
            )

        return "\n".join(plan)


study_planner = StudyPlannerTool()


# =========================================================
# CURRENT SUBJECT
# =========================================================

def get_subject():

    if st.session_state.subject_option == "➕ Add Your Own Subject":

        custom = st.session_state.custom_subject.strip()

        if custom:
            return custom

        return "Custom Subject"

    return st.session_state.subject_option


# =========================================================
# CREATE LLM
# =========================================================

def create_llm():

    api_key = st.secrets.get("GROQ_API_KEY")

    if not api_key:

        st.error(
            "GROQ_API_KEY was not found. "
            "Please add it in Streamlit Secrets."
        )

        st.stop()

    return LLM(
        model="groq/openai/gpt-oss-120b",
        api_key=api_key,
        temperature=0.3,
        reasoning_effort="medium",
    )


# =========================================================
# CREATE TUTOR AGENT
# =========================================================

def create_tutor_agent():

    llm = create_llm()

    agent = Agent(
        role="Personal AI Study Tutor",
        goal=(
            "Help students understand academic subjects clearly, "
            "practice effectively, take quizzes, and create "
            "realistic study plans."
        ),
        backstory=(
            "You are a patient university-level AI tutor. "
            "You explain difficult concepts in simple language "
            "using examples, tables, step-by-step explanations, "
            "and small diagrams when useful."
        ),
        tools=[
            calculator,
            study_planner,
        ],
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )

    return agent


# =========================================================
# MODE INSTRUCTIONS
# =========================================================

def get_mode_instructions(mode):

    if mode == "Learn":

        return """
You are in LEARN mode.

Teach the student clearly and step-by-step.

Use:
- Simple explanations
- University-level accuracy
- Real-world examples
- Tables when useful
- Small text diagrams when useful
- Important terms and definitions
- A short summary at the end

Do not make the explanation unnecessarily complicated.
"""

    if mode == "Practice":

        return """
You are in PRACTICE mode.

Help the student actively practice.

Give:
- Practice questions
- Coding/problem-solving exercises
- Scenario-based questions
- Gradually increasing difficulty

If the student asks for questions without answers,
do not reveal the answers immediately.

Give hints when requested.
After the student attempts an answer, explain the solution.
"""

    if mode == "Quiz":

        return """
You are in QUIZ mode.

Act like an interactive tutor quiz.

Rules:
- Ask one question at a time unless the student asks for multiple.
- Use MCQs, conceptual questions, and scenario-based questions.
- Wait for the student's answer.
- Then tell them whether it is correct.
- Explain why.
"""

    return """
You are in STUDY PLAN mode.

Create realistic and manageable study plans.

Consider:
- Subject
- Topic
- Student level
- Available study time
- Number of days

Divide work into daily tasks.

Include:
- Learning
- Practice
- Revision
- Short breaks when appropriate

Do not create an unrealistic workload.
"""


# =========================================================
# ASK TUTOR
# =========================================================

def ask_tutor(question):

    agent = create_tutor_agent()

    subject = get_subject()

    mode_instructions = get_mode_instructions(
        st.session_state.mode
    )

    history_text = ""

    for message in st.session_state.messages[-10:]:

        role = message.get("role", "")
        content = message.get("content", "")

        history_text += f"{role}: {content}\n"

    prompt = f"""
You are the student's personal AI study tutor.

STUDENT PROFILE

Name: {st.session_state.student_name}
Subject: {subject}
Current Topic: {st.session_state.current_topic}
Learning Level: {st.session_state.learning_level}
Current Mode: {st.session_state.mode}

MODE INSTRUCTIONS

{mode_instructions}

CONVERSATION HISTORY

{history_text}

STUDENT'S NEW QUESTION

{question}

IMPORTANT RULES

1. Address the student's actual question.
2. Match the student's learning level.
3. Use simple English.
4. Explain technical terms clearly.
5. Use examples whenever useful.
6. If mathematics is needed, use the calculator tool.
7. If a study plan is needed, use the Study Planner tool when appropriate.
8. Keep answers organized and readable.
9. Encourage understanding rather than memorization.
"""

    task = Task(
        description=prompt,
        expected_output=(
            "A clear, accurate, helpful response suitable "
            "for a university student."
        ),
        agent=agent,
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=False,
    )

    result = crew.kickoff()

    return str(result)


# =========================================================
# WELCOME SCREEN
# =========================================================

def show_welcome_screen():

    st.markdown(
        '<div class="main-title">🎓 Study Tutor Agent</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        'Your Personal AI Study Assistant'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="welcome-card">

        <h2>👋 Welcome to Study Tutor!</h2>

        <p>
        Your AI study assistant can help you learn concepts,
        practice questions, take quizzes, and create study plans.
        </p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("✨ What can you do here?")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown(
            """
            <div class="feature-card">
            <h3>📚 Learn</h3>
            <p>
            Understand difficult concepts
            step-by-step with examples.
            </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:

        st.markdown(
            """
            <div class="feature-card">
            <h3>🧠 Practice</h3>
            <p>
            Practice questions and
            problem-solving exercises.
            </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:

        st.markdown(
            """
            <div class="feature-card">
            <h3>📝 Quiz</h3>
            <p>
            Test your knowledge with
            interactive questions.
            </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:

        st.markdown(
            """
            <div class="feature-card">
            <h3>📅 Study Plan</h3>
            <p>
            Create a realistic plan
            for your learning goals.
            </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    st.subheader("🚀 Let's personalize your learning")

    st.info(
        "Enter your information below. "
        "You can change it later."
    )

    with st.form("onboarding_form"):

        name = st.text_input(
            "👩‍🎓 Your Name",
            placeholder="Enter your name",
        )

        subject_option = st.selectbox(
            "📚 Choose Your Subject",
            SUBJECTS,
        )

        custom_subject = ""

        if subject_option == "➕ Add Your Own Subject":

            custom_subject = st.text_input(
                "✏️ Enter Your Subject",
                placeholder="e.g. Digital Logic Design",
            )

        topic = st.text_input(
            "🎯 What are you studying?",
            placeholder="e.g. Queue, HTML Forms, RSA, Python",
        )

        level = st.selectbox(
            "📈 Your Learning Level",
            [
                "Beginner",
                "Intermediate",
                "Advanced",
            ],
        )

        start = st.form_submit_button(
            "🚀 Start Learning",
            use_container_width=True,
            type="primary",
        )

        if start:

            if not name.strip():

                st.error("Please enter your name.")

            elif (
                subject_option == "➕ Add Your Own Subject"
                and not custom_subject.strip()
            ):

                st.error("Please enter your custom subject.")

            elif not topic.strip():

                st.error(
                    "Please enter the topic you want to study."
                )

            else:

                st.session_state.student_name = name.strip()

                st.session_state.subject_option = (
                    subject_option
                )

                st.session_state.custom_subject = (
                    custom_subject.strip()
                )

                st.session_state.current_topic = topic.strip()

                st.session_state.learning_level = level

                st.session_state.mode = "Learn"

                st.session_state.messages = []

                st.session_state.previous_subject = get_subject()

                st.session_state.previous_topic = topic.strip()

                st.session_state.setup_complete = True

                st.rerun()


# =========================================================
# FIRST OPEN
# =========================================================

if not st.session_state.setup_complete:

    show_welcome_screen()

    st.stop()


# =========================================================
# CURRENT SUBJECT
# =========================================================

current_subject = get_subject()


# =========================================================
# DETECT TOPIC / SUBJECT CHANGE
# =========================================================

if (
    st.session_state.previous_subject
    and (
        current_subject != st.session_state.previous_subject
        or st.session_state.current_topic
        != st.session_state.previous_topic
    )
):

    st.session_state.messages = []

    st.session_state.previous_subject = current_subject

    st.session_state.previous_topic = (
        st.session_state.current_topic
    )


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🎓 Study Tutor Agent</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Your Personal AI Study Assistant'
    '</div>',
    unsafe_allow_html=True,
)


# =========================================================
# TOP NAVIGATION
# =========================================================

nav1, nav2, nav3, nav4 = st.columns(4)

with nav1:

    if st.button(
        "📚 Learn",
        use_container_width=True,
    ):

        st.session_state.mode = "Learn"
        st.rerun()


with nav2:

    if st.button(
        "🧠 Practice",
        use_container_width=True,
    ):

        st.session_state.mode = "Practice"
        st.rerun()


with nav3:

    if st.button(
        "📝 Quiz",
        use_container_width=True,
    ):

        st.session_state.mode = "Quiz"
        st.rerun()


with nav4:

    if st.button(
        "📅 Study Plan",
        use_container_width=True,
    ):

        st.session_state.mode = "Study Plan"
        st.rerun()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("👩‍🎓 Student Profile")

    st.markdown(
        f"""
        <div class="profile-box">

        <b>Name:</b> {st.session_state.student_name}<br>

        <b>Subject:</b> {current_subject}<br>

        <b>Topic:</b> {st.session_state.current_topic}<br>

        <b>Level:</b> {st.session_state.learning_level}

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.subheader("⚙️ Learning Setup")

    subject_index = 0

    if st.session_state.subject_option in SUBJECTS:

        subject_index = SUBJECTS.index(
            st.session_state.subject_option
        )

    new_subject_option = st.selectbox(
        "📚 Subject",
        SUBJECTS,
        index=subject_index,
        key="sidebar_subject",
    )

    new_custom_subject = ""

    if new_subject_option == "➕ Add Your Own Subject":

        new_custom_subject = st.text_input(
            "✏️ Custom Subject",
            value=st.session_state.custom_subject,
            key="sidebar_custom_subject",
        )

    new_topic = st.text_input(
        "🎯 Current Topic",
        value=st.session_state.current_topic,
        key="sidebar_topic",
    )

    level_options = [
        "Beginner",
        "Intermediate",
        "Advanced",
    ]

    level_index = level_options.index(
        st.session_state.learning_level
    )

    new_level = st.selectbox(
        "📈 Learning Level",
        level_options,
        index=level_index,
        key="sidebar_level",
    )

    # -----------------------------------------------------
    # APPLY CHANGES
    # -----------------------------------------------------

    if st.button(
        "💾 Apply Changes",
        use_container_width=True,
    ):

        old_subject = get_subject()
        old_topic = st.session_state.current_topic

        st.
