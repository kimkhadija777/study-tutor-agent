import streamlit as st
import ast
import operator

from crewai import Agent, Task, Crew, LLM
from crewai.tools import BaseTool


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="Study Tutor Agent",
    page_icon="🎓",
    layout="wide"
)


# =========================================================
# STYLE
# =========================================================

st.markdown(
    """
    <style>
    .title {
        font-size: 40px;
        font-weight: 700;
    }

    .subtitle {
        font-size: 18px;
        color: #666666;
        margin-bottom: 25px;
    }

    .welcome {
        padding: 30px;
        border-radius: 18px;
        background: #f5f7ff;
        border: 1px solid #dfe4ff;
        margin-bottom: 25px;
    }

    .feature {
        padding: 18px;
        border-radius: 14px;
        border: 1px solid #dddddd;
        min-height: 140px;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# CREWAI COMPATIBILITY FIX
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

if "subject" not in st.session_state:
    st.session_state.subject = "Computer Science"

if "custom_subject" not in st.session_state:
    st.session_state.custom_subject = ""

if "topic" not in st.session_state:
    st.session_state.topic = ""

if "level" not in st.session_state:
    st.session_state.level = "Beginner"

if "mode" not in st.session_state:
    st.session_state.mode = "Learn"

if "messages" not in st.session_state:
    st.session_state.messages = []


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
    "➕ Add Your Own Subject"
]


# =========================================================
# GET SUBJECT
# =========================================================

def get_subject():

    if st.session_state.subject == "➕ Add Your Own Subject":

        if st.session_state.custom_subject.strip():
            return st.session_state.custom_subject.strip()

        return "Custom Subject"

    return st.session_state.subject


# =========================================================
# CALCULATOR
# =========================================================

OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos
}


def calculate(expression):

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

                operation = OPERATORS.get(type(node.op))

                if operation is None:
                    raise ValueError("Operator not allowed")

                return operation(left, right)

            if isinstance(node, ast.UnaryOp):

                value = evaluate(node.operand)

                operation = OPERATORS.get(type(node.op))

                if operation is None:
                    raise ValueError("Operator not allowed")

                return operation(value)

            raise ValueError("Invalid expression")

        return evaluate(tree.body)

    except Exception as error:

        return f"Error: {error}"


class CalculatorTool(BaseTool):

    name: str = "Calculator"

    description: str = (
        "Use this tool for mathematical calculations."
    )

    def _run(self, expression: str) -> str:

        return str(calculate(expression))


calculator = CalculatorTool()


# =========================================================
# STUDY PLANNER
# =========================================================

class StudyPlannerTool(BaseTool):

    name: str = "Study Planner"

    description: str = (
        "Create a simple study plan for a student."
    )

    def _run(
        self,
        subject: str,
        topics: str,
        days: int,
        hours_per_day: float
    ) -> str:

        topic_list = [
            x.strip()
            for x in topics.split(",")
            if x.strip()
        ]

        if not topic_list:
            return "Please provide topics."

        result = []

        for day in range(1, days + 1):

            topic = topic_list[(day - 1) % len(topic_list)]

            result.append(
                f"Day {day}: {topic} - "
                f"{hours_per_day} hour(s)"
            )

        return "\n".join(result)


study_planner = StudyPlannerTool()


# =========================================================
# LLM
# =========================================================

def create_llm():

    api_key = st.secrets.get("GROQ_API_KEY")

    if not api_key:

        st.error(
            "GROQ_API_KEY is missing. "
            "Please add it to Streamlit Secrets."
        )

        st.stop()

    return LLM(
        model="groq/openai/gpt-oss-120b",
        api_key=api_key,
        temperature=0.3,
        reasoning_effort="medium"
    )


# =========================================================
# AGENT
# =========================================================

def create_agent():

    agent = Agent(
        role="Personal AI Study Tutor",
        goal=(
            "Help students understand subjects, practice, "
            "take quizzes, and create study plans."
        ),
        backstory=(
            "You are a patient university-level tutor. "
            "Explain difficult concepts in simple language "
            "with examples and step-by-step explanations."
        ),
        tools=[
            calculator,
            study_planner
        ],
        llm=create_llm(),
        allow_delegation=False,
        verbose=False
    )

    return agent


# =========================================================
# MODE INSTRUCTIONS
# =========================================================

def mode_instructions():

    mode = st.session_state.mode

    if mode == "Learn":

        return """
Teach the student step-by-step.
Use simple language, examples, definitions,
tables, and small diagrams when useful.
End with a short summary.
"""

    if mode == "Practice":

        return """
Help the student practice.
Give exercises and questions.
If the student asks for questions without answers,
do not reveal the answers immediately.
Give hints when requested.
"""

    if mode == "Quiz":

        return """
Act as an interactive quiz tutor.
Ask one question at a time.
Wait for the student's answer.
Then evaluate it and explain why.
"""

    return """
Create realistic study plans.
Include learning, practice,
revision, and reasonable breaks.
Do not create an unrealistic workload.
"""


# =========================================================
# ASK AI
# =========================================================

def ask_tutor(question):

    agent = create_agent()

    history = ""

    for message in st.session_state.messages[-10:]:

        history += (
            message["role"]
            + ": "
            + message["content"]
            + "\n"
        )

    prompt = f"""
You are a personal AI study tutor.

Student name:
{st.session_state.student_name}

Subject:
{get_subject()}

Current topic:
{st.session_state.topic}

Learning level:
{st.session_state.level}

Current mode:
{st.session_state.mode}

Mode instructions:
{mode_instructions()}

Recent conversation:
{history}

Student question:
{question}

Rules:
- Answer the actual question.
- Match the student's level.
- Use simple English.
- Explain technical terms.
- Give examples when useful.
- Be accurate.
- Keep the answer organized.
"""

    task = Task(
        description=prompt,
        expected_output="A clear and helpful study response.",
        agent=agent
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=False
    )

    result = crew.kickoff()

    return str(result)


# =========================================================
# WELCOME SCREEN
# =========================================================

def welcome_screen():

    st.markdown(
        '<div class="title">🎓 Study Tutor Agent</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Your Personal AI Study Assistant'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="welcome">

        <h2>👋 Welcome to Study Tutor!</h2>

        <p>
        Your AI study assistant can help you learn,
        practice, take quizzes, and create study plans.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("✨ What can I help you with?")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown(
            """
            <div class="feature">
            <h3>📚 Learn</h3>
            <p>
            Understand concepts step-by-step
            with simple explanations.
            </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class="feature">
            <h3>🧠 Practice</h3>
            <p>
            Practice questions and
            problem-solving exercises.
            </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            """
            <div class="feature">
            <h3>📝 Quiz</h3>
            <p>
            Test your knowledge with
            interactive quizzes.
            </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:

        st.markdown(
            """
            <div class="feature">
            <h3>📅 Study Plan</h3>
            <p>
            Create a personalized study
            schedule.
            </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    st.subheader("🚀 Let's get started")

    st.info(
        "Tell me a little about what you want to learn."
    )

    with st.form("setup_form"):

        name = st.text_input(
            "👩‍🎓 Your Name",
            placeholder="e.g. Khadija"
        )

        subject = st.selectbox(
            "📚 Choose Your Subject",
            SUBJECTS
        )

        custom_subject = ""

        if subject == "➕ Add Your Own Subject":

            custom_subject = st.text_input(
                "✏️ Your Subject",
                placeholder="e.g. Digital Logic Design"
            )

        topic = st.text_input(
            "🎯 What do you want to study?",
            placeholder="e.g. Queue"
        )

        level = st.selectbox(
            "📈 Your Learning Level",
            [
                "Beginner",
                "Intermediate",
                "Advanced"
            ]
        )

        start = st.form_submit_button(
            "🚀 Start Learning",
            use_container_width=True,
            type="primary"
        )

        if start:

            if not name.strip():

                st.error("Please enter your name.")

            elif (
                subject == "➕ Add Your Own Subject"
                and not custom_subject.strip()
            ):

                st.error("Please enter your subject.")

            elif not topic.strip():

                st.error("Please enter your topic.")

            else:

                st.session_state.student_name = name.strip()

                st.session_state.subject = subject

                st.session_state.custom_subject = (
                    custom_subject.strip()
                )

                st.session_state.topic = topic.strip()

                st.session_state.level = level

                st.session_state.mode = "Learn"

                st.session_state.messages = []

                st.session_state.setup_complete = True

                st.rerun()


# =========================================================
# SHOW WELCOME FIRST
# =========================================================

if not st.session_state.setup_complete:

    welcome_screen()

    st.stop()


# =========================================================
# MAIN HEADER
# =========================================================

st.markdown(
    '<div class="title">🎓 Study Tutor Agent</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Your Personal AI Study Assistant'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# NAVIGATION
# =========================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    if st.button(
        "📚 Learn",
        use_container_width=True
    ):

        st.session_state.mode = "Learn"
        st.rerun()

with col2:

    if st.button(
        "🧠 Practice",
        use_container_width=True
    ):

        st.session_state.mode = "Practice"
        st.rerun()

with col3:

    if st.button(
        "📝 Quiz",
        use_container_width=True
    ):

        st.session_state.mode = "Quiz"
        st.rerun()

with col4:

    if st.button(
        "📅 Study Plan",
        use_container_width=True
    ):

        st.session_state.mode = "Study Plan"
        st.rerun()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("👩‍🎓 Student Profile")

    st.write(
        f"**Name:** {st.session_state.student_name}"
    )

    st.write(
        f"**Subject:** {get_subject()}"
    )

    st.write(
        f"**Topic:** {st.session_state.topic}"
    )

    st.write(
        f"**Level:** {st.session_state.level}"
    )

    st.markdown("---")

    st.subheader("⚙️ Learning Setup")

    selected_subject = st.selectbox(
        "📚 Subject",
        SUBJECTS,
        index=SUBJECTS.index(st.session_state.subject),
        key="change_subject"
    )

    selected_custom = ""

    if selected_subject == "➕ Add Your Own Subject":

        selected_custom = st.text_input(
            "✏️ Custom Subject",
            value=st.session_state.custom_subject
        )

    selected_topic = st.text_input(
        "🎯 Topic",
        value=st.session_state.topic
    )

    selected_level = st.selectbox(
        "📈 Level",
        [
            "Beginner",
            "Intermediate",
            "Advanced"
        ],
        index=[
            "Beginner",
            "Intermediate",
            "Advanced"
        ].index(st.session_state.level)
    )

    if st.button(
        "💾 Apply Changes",
        use_container_width=True
    ):

        old_subject = get_subject()
        old_topic = st.session_state.topic

        st.session_state.subject = selected_subject

        st.session_state.custom_subject = selected_custom

        st.session_state.topic = selected_topic

        st.session_state.level = selected_level

        new_subject = get_subject()

        if (
            old_subject != new_subject
            or old_topic != selected_topic
        ):

            st.session_state.messages = []

        st.success("Learning setup updated!")

        st.rerun()

    if st.button(
        "🧹 Clear Conversation",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()

    if st.button(
        "⚙️ Edit Welcome Setup",
        use_container_width=True
    ):

        st.session_state.setup_complete = False

        st.rerun()


# =========================================================
# CURRENT INFORMATION
# =========================================================

st.info(
    f"Mode: {st.session_state.mode}  |  "
    f"Subject: {get_subject()}  |  "
    f"Topic: {st.session_state.topic}"
)


# =========================================================
# SUGGESTED QUESTIONS
# =========================================================

st.subheader("💡 Try asking")

if st.session_state.mode == "Learn":

    suggestions = [
        "Explain this topic in simple words.",
        "Give me a real-world example.",
        "Explain it with a small diagram."
    ]

elif st.session_state.mode == "Practice":

    suggestions = [
        "Give me 5 practice questions.",
        "Give me a problem-solving exercise.",
        "Give me a scenario-based question."
    ]

elif st.session_state.mode == "Quiz":

    suggestions = [
        "Start a quiz. Ask one question at a time.",
        "Give me 5 MCQs.",
        "Give me a difficult quiz."
    ]

else:

    suggestions = [
        "Make me a 7-day study plan.",
        "I can study 2 hours per day.",
        "Include practice and revision."
    ]


a, b, c = st.columns(3)

with a:

    if st.button(
        suggestions[0],
        use_container_width=True
    ):

        question = suggestions[0]

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.spinner("🤖 Thinking..."):

            try:

                answer = ask_tutor(question)

            except Exception as error:

                answer = f"Error: {error}"

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        st.rerun()


with b:

    if st.button(
        suggestions[1],
        use_container_width=True
    ):

        question = suggestions[1]

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.spinner("🤖 Thinking..."):

            try:

                answer = ask_tutor(question)

            except Exception as error:

                answer = f"Error: {error}"

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        st.rerun()


with c:

    if st.button(
        suggestions[2],
        use_container_width=True
    ):

        question = suggestions[2]

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.spinner("🤖 Thinking..."):

            try:

                answer = ask_tutor(question)

            except Exception as error:

                answer = f"Error: {error}"

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        st.rerun()


# =========================================================
# CHAT HISTORY
# ==============================
