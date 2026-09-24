import ast
import operator as op
import streamlit as st

from crewai import Agent, LLM
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
# CACHE WORKAROUND
# =========================================================

try:
    import crewai.llms.cache as crewai_cache
    crewai_cache.mark_cache_breakpoint = lambda message: message
except Exception:
    pass


# =========================================================
# CSS
# =========================================================

st.markdown("""
<style>
.hero {
    padding: 20px;
    border-radius: 15px;
    background: rgba(100,100,100,0.08);
    margin-bottom: 20px;
}

.card {
    padding: 15px;
    border: 1px solid rgba(128,128,128,0.25);
    border-radius: 12px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# SESSION STATE
# =========================================================

defaults = {
    "setup_complete": False,
    "student_name": "",
    "subject": "Data Structures",
    "custom_subject": "",
    "topic": "",
    "level": "Beginner",
    "mode": "Learn",
    "messages": [],
    "pending_question": ""
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


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


def get_subject():
    if st.session_state.subject == "➕ Add Your Own Subject":
        if st.session_state.custom_subject.strip():
            return st.session_state.custom_subject.strip()
        return "Your Subject"

    return st.session_state.subject


# =========================================================
# CALCULATOR
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
                return OPERATORS[type(node.op)](
                    evaluate(node.operand)
                )

        raise ValueError("Invalid calculation.")

    tree = ast.parse(expression, mode="eval")
    return evaluate(tree.body)


class CalculatorTool(BaseTool):

    name: str = "Calculator"

    description: str = (
        "Calculate basic mathematical expressions."
    )

    def _run(self, expression: str) -> str:

        try:
            result = calculate(expression)
            return f"Result: {result}"

        except Exception as error:
            return f"Calculator error: {error}"


# =========================================================
# STUDY PLANNER
# =========================================================

class StudyPlannerTool(BaseTool):

    name: str = "Study Planner"

    description: str = (
        "Help create practical study plans."
    )

    def _run(self, request: str) -> str:

        return (
            "Create a realistic study plan based on the "
            "student's requested subjects, time, and goals."
        )


calculator = CalculatorTool()
study_planner = StudyPlannerTool()


# =========================================================
# LLM
# =========================================================

def create_llm():

    api_key = st.secrets.get("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing from Streamlit Secrets."
        )

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

    return Agent(
        role="Personal Study Tutor",

        goal=(
            "Help a university student understand concepts "
            "clearly and improve through practice."
        ),

        backstory=(
            "You are a friendly and patient university tutor. "
            "Explain difficult concepts in simple language. "
            "Use examples, tables, diagrams, and code when useful."
        ),

        llm=create_llm(),

        tools=[
            calculator,
            study_planner
        ],

        allow_delegation=False,

        verbose=False
    )


# =========================================================
# MODE
# =========================================================

def get_mode_instruction():

    mode = st.session_state.mode

    if mode == "Learn":
        return (
            "Teach the concept clearly from basics. "
            "Use examples and simple explanations."
        )

    if mode == "Practice":
        return (
            "Help the student practice. Give questions, "
            "hints, and explanations. Do not give answers "
            "when the student specifically asks for questions only."
        )

    if mode == "Quiz":
        return (
            "Act as a quiz tutor. Ask questions and explain "
            "the student's answers."
        )

    if mode == "Study Plan":
        return (
            "Create a realistic study plan with study blocks "
            "and short breaks."
        )

    return "Teach the topic clearly."


# =========================================================
# ASK AI
# =========================================================

def ask_tutor(question):

    history = st.session_state.messages[-8:]

    history_text = "\n".join(
        f"{item['role'].upper()}: {item['content']}"
        for item in history
    )

    prompt = f"""
You are a personal AI study tutor.

Student:
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
{get_mode_instruction()}

Teaching rules:

- Use simple clear English.
- Explain technical words.
- Give practical examples.
- Keep answers organized.
- Use tables when useful.
- Use small diagrams when useful.
- For programming questions, provide correct code.
- Adapt the explanation to the student's level.
- Do not unnecessarily make answers extremely long.

Previous conversation:
{history_text if history_text else "No previous conversation."}

New student question:
{question}

Answer the student directly.
"""

    agent = create_agent()

    result = agent.kickoff(prompt)

    return str(result)


# =========================================================
# WELCOME SCREEN
# =========================================================

def show_welcome():

    st.markdown(
        """
        <div class="hero">
        <h1>🎓 Welcome to Study Tutor Agent</h1>
        <p>Your Personal AI Study Assistant</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("✨ What can your tutor do?")

    columns = st.columns(4)

    features = [
        (
            "📚 Learn",
            "Understand concepts with simple explanations."
        ),
        (
            "🧠 Practice",
            "Practice with questions and hints."
        ),
        (
            "📝 Quiz",
            "Test your knowledge."
        ),
        (
            "📅 Study Plan",
            "Create a practical study schedule."
        )
    ]

    for column, feature in zip(columns, features):

        with column:
            st.markdown(
                f"""
                <div class="card">
                <h3>{feature[0]}</h3>
                <p>{feature[1]}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("---")

    st.subheader("👩‍🎓 Set Up Your Learning")

    with st.form("setup_form"):

        name = st.text_input(
            "Your Name",
            placeholder="e.g. Khadija"
        )

        subject = st.selectbox(
            "📚 Subject",
            SUBJECTS
        )

        custom_subject = ""

        if subject == "➕ Add Your Own Subject":

            custom_subject = st.text_input(
                "Your Subject",
                placeholder="e.g. Digital Logic Design"
            )

        topic = st.text_input(
            "🎯 Current Topic",
            placeholder="e.g. Queue"
        )

        level = st.selectbox(
            "📈 Learning Level",
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
            return

        if subject == "➕ Add Your Own Subject":
            if not custom_subject.strip():
                st.error("Please enter your subject.")
                return

        if not topic.strip():
            st.error("Please enter your topic.")
            return

        st.session_state.student_name = name.strip()
        st.session_state.subject = subject
        st.session_state.custom_subject = custom_subject.strip()
        st.session_state.topic = topic.strip()
        st.session_state.level = level
        st.session_state.mode = "Learn"
        st.session_state.messages = []
        st.session_state.setup_complete = True

        st.rerun()


# =========================================================
# FIRST SCREEN
# =========================================================

if not st.session_state.setup_complete:

    show_welcome()

    st.stop()


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">
    <h1>🎓 Study Tutor Agent</h1>
    <p>Your Personal AI Study Assistant</p>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# NAVIGATION
# =========================================================

st.subheader("Choose Your Learning Mode")

nav1, nav2, nav3, nav4 = st.columns(4)

if nav1.button(
    "📚 Learn",
    use_container_width=True
):
    st.session_state.mode = "Learn"

if nav2.button(
    "🧠 Practice",
    use_container_width=True
):
    st.session_state.mode = "Practice"

if nav3.button(
    "📝 Quiz",
    use_container_width=True
):
    st.session_state.mode = "Quiz"

if nav4.button(
    "📅 Study Plan",
    use_container_width=True
):
    st.session_state.mode = "Study Plan"


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

    st.header("⚙️ Learning Setup")

    sidebar_subject = st.selectbox(
        "📚 Subject",
        SUBJECTS,
        index=SUBJECTS.index(
            st.session_state.subject
        )
    )

    sidebar_custom = st.session_state.custom_subject

    if sidebar_subject == "➕ Add Your Own Subject":

        sidebar_custom = st.text_input(
            "Your Subject",
            value=st.session_state.custom_subject
        )

    sidebar_topic = st.text_input(
        "🎯 Topic",
        value=st.session_state.topic
    )

    levels = [
        "Beginner",
        "Intermediate",
        "Advanced"
    ]

    sidebar_level = st.selectbox(
        "📈 Level",
        levels,
        index=levels.index(
            st.session_state.level
        )
    )

    if st.button(
        "💾 Apply Changes",
        use_container_width=True
    ):

        st.session_state.subject = sidebar_subject
        st.session_state.custom_subject = sidebar_custom
        st.session_state.topic = sidebar_topic
        st.session_state.level = sidebar_level

        st.session_state.messages = []

        st.success("Learning setup updated!")

        st.rerun()

    if st.button(
        "✏️ Edit Welcome Setup",
        use_container_width=True
    ):

        st.session_state.setup_complete = False
        st.rerun()

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.success("Conversation cleared!")

        st.rerun()


# =========================================================
# SESSION INFO
# =========================================================

st.info(
    f"**Mode:** {st.session_state.mode}  |  "
    f"**Subject:** {get_subject()}  |  "
    f"**Topic:** {st.session_state.topic}"
)


# =========================================================
# SUGGESTIONS
# =========================================================

st.subheader("💡 Try Asking")

topic = st.session_state.topic

if st.session_state.mode == "Learn":

    suggestions = [
        f"Explain {topic} in simple words.",
        f"Give me an example of {topic}.",
        f"Explain {topic} with a small diagram."
    ]

elif st.session_state.mode == "Practice":

    suggestions = [
        f"Give me 5 practice questions about {topic}.",
        f"Give me one question about {topic}.",
        f"Give me a coding problem about {topic}."
    ]

elif st.session_state.mode == "Quiz":

    suggestions = [
        f"Start a quiz about {topic}.",
        f"Give me an MCQ about {topic}.",
        "Explain my answer after I respond."
    ]

else:

    suggestions = [
        f"Make me a 7-day plan for {topic}.",
        "Make me a 2-hour daily study plan.",
        f"How should I revise {topic}?"
    ]


suggestion_columns = st.columns(3)

for i, suggestion in enumerate(suggestions):

    if suggestion_columns[i].button(
        suggestion,
        key=f"suggestion_{i}",
        use_container_width=True
    ):

        st.session_state.pending_question = suggestion


# =========================================================
# ASK TUTOR
# =========================================================

st.markdown("---")

st.subheader("💬 Ask Your AI Tutor")

st.write(
    f"Ask anything about **{topic}**. "
    "Your tutor will explain it according to your level."
)


question_default = st.session_state.pending_question

st.session_state.pending_question = ""


question = st.text_area(
    "Your Question",
    value=question_default,
    placeholder=(
        "For example: Explain Queue in simple words "
        "with an example and diagram."
    ),
    height=130
)


if st.button(
    "🚀 Ask Tutor",
    use_container_width=True,
    type="primary"
):

    if not question.strip():

        st.warning("Please type a question first.")

    else:

        user_question = question.strip()

        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_question
            }
        )

        with st.spinner(
            "🤖 Study Tutor is thinking..."
        ):

            try:

                answer = ask_tutor(
                    user_question
                )

            except Exception as error:

                answer = (
                    "Sorry, something went wrong.\n\n"
                    f"Error: {error}"
                )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        st.rerun()


# =========================================================
# CONVERSATION
# =========================================================

if st.session_state.messages:

    st.markdown("---")

    st.subheader("💬 Conversation")

    for message in st.session_state.messages:

        role = message["role"]

        with st.chat_message(role):

            st.markdown(
                message["content"]
            )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "🎓 Study Tutor Agent | Learn smarter, practice better."
)
