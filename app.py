import ast
import operator as op
import streamlit as st

st.set_page_config(
    page_title="Study Tutor Agent",
    page_icon="🎓",
    layout="wide"
)

# =========================================================
# CSS
# =========================================================

st.markdown("""
<style>
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.hero {
    padding: 1.2rem;
    border-radius: 16px;
    background: rgba(100, 100, 100, 0.08);
    margin-bottom: 1rem;
}

.card {
    padding: 1rem;
    border: 1px solid rgba(128, 128, 128, 0.25);
    border-radius: 14px;
    margin-bottom: 0.7rem;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# CREWAI / GROQ CACHE COMPATIBILITY WORKAROUND
# =========================================================

try:
    import crewai.llms.cache as crewai_cache
    crewai_cache.mark_cache_breakpoint = lambda message: message
except Exception:
    pass


from crewai import Agent, LLM
from crewai.tools import BaseTool


# =========================================================
# SESSION STATE
# =========================================================

def init_state():

    defaults = {
        "setup_complete": False,
        "student_name": "",
        "subject": "Data Structures",
        "custom_subject": "",
        "topic": "",
        "level": "Beginner",
        "mode": "Learn",
        "messages": [],
    }

    for key, value in defaults.items():

        if key not in st.session_state:
            st.session_state[key] = value


init_state()


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


def get_subject():

    if st.session_state.subject == "➕ Add Your Own Subject":

        custom = st.session_state.custom_subject.strip()

        return custom if custom else "Your Subject"

    return st.session_state.subject


# =========================================================
# SAFE CALCULATOR
# =========================================================

ALLOWED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}


def safe_calculate(expression):

    def evaluate(node):

        if isinstance(node, ast.Constant) and isinstance(
            node.value,
            (int, float)
        ):
            return node.value

        if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_OPERATORS:

            left = evaluate(node.left)
            right = evaluate(node.right)

            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("Exponent is too large.")

            return ALLOWED_OPERATORS[type(node.op)](
                left,
                right
            )

        if isinstance(node, ast.UnaryOp) and type(node.op) in ALLOWED_OPERATORS:

            return ALLOWED_OPERATORS[type(node.op)](
                evaluate(node.operand)
            )

        raise ValueError(
            "Only basic arithmetic expressions are allowed."
        )

    tree = ast.parse(
        expression,
        mode="eval"
    )

    return evaluate(tree.body)


class CalculatorTool(BaseTool):

    name: str = "Calculator"

    description: str = (
        "Calculate basic arithmetic expressions such as "
        "25*4, 100/5, or (10+5)*2."
    )

    def _run(self, expression: str) -> str:

        try:

            result = safe_calculate(expression)

            return f"Result: {result}"

        except Exception as error:

            return f"Calculator error: {error}"


# =========================================================
# STUDY PLANNER TOOL
# =========================================================

class StudyPlannerTool(BaseTool):

    name: str = "Study Planner"

    description: str = (
        "Create a practical study plan when the student gives "
        "subjects/topics, available time, and duration."
    )

    def _run(self, request: str) -> str:

        return (
            "Create the study plan directly from the student's request. "
            "Divide available time into realistic study blocks, include "
            "short breaks, and prioritize the most important topics first."
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
            "GROQ_API_KEY is missing. Add it in "
            "Streamlit Cloud > Settings > Secrets."
        )

    return LLM(
        model="groq/openai/gpt-oss-120b",
        api_key=api_key,
        temperature=0.3,
        reasoning_effort="medium",
    )


# =========================================================
# AGENT
# =========================================================

def create_agent():

    return Agent(

        role="Personal Study Tutor",

        goal=(
            "Help a university student understand concepts clearly "
            "and improve through practice."
        ),

        backstory=(
            "You are a patient, friendly university-level AI tutor. "
            "You explain difficult concepts in simple language, use "
            "examples, tables, small diagrams when useful, and adapt "
            "explanations to the student's level."
        ),

        llm=create_llm(),

        tools=[
            calculator,
            study_planner
        ],

        allow_delegation=False,

        verbose=False,
    )


# =========================================================
# MODE INSTRUCTIONS
# =========================================================

def mode_instructions(mode):

    instructions = {

        "Learn": (
            "Teach the concept clearly. Start from basics, "
            "then give an example. Use a small table or diagram "
            "when useful."
        ),

        "Practice": (
            "Help the student practice. Give questions, examples, "
            "hints, and explanations. If the student asks for "
            "questions without answers, do not reveal the answers."
        ),

        "Quiz": (
            "Act as a quiz tutor. Give one question at a time "
            "when appropriate, wait for the student's answer, "
            "then explain whether it is correct and why."
        ),

        "Study Plan": (
            "Create a realistic study plan based on the student's "
            "available time and goals. Break it into manageable "
            "sessions and include short breaks."
        ),
    }

    return instructions.get(
        mode,
        instructions["Learn"]
    )


# =========================================================
# ASK TUTOR
# =========================================================

def ask_tutor(question):

    subject = get_subject()

    topic = (
        st.session_state.topic.strip()
        or "General"
    )

    level = st.session_state.level

    mode = st.session_state.mode

    name = (
        st.session_state.student_name.strip()
        or "Student"
    )

    recent_messages = st.session_state.messages[-8:]

    history_text = "\n".join(
        f"{message['role'].upper()}: {message['content']}"
        for message in recent_messages
    )

    prompt = f"""
You are the student's personal AI study tutor.

Student name: {name}

Subject: {subject}

Current topic: {topic}

Learning level: {level}

Current mode: {mode}


Mode instructions:

{mode_instructions(mode)}


Important teaching rules:

- Explain at a university beginner/intermediate level unless the student asks for advanced detail.
- Use simple, clear English.
- Give concrete examples.
- Avoid unnecessary jargon; define technical terms when you use them.
- Be encouraging but do not make the answer overly long.
- For programming questions, show correct code when useful and explain it simply.
- For calculations, use the calculator tool when appropriate.
- For study-plan requests, use the Study Planner tool when appropriate.
- Stay focused on the student's learning request.


Recent conversation:

{history_text if history_text else "No previous conversation."}


Student's new question:

{question}


Answer the student's question directly.
"""

    agent = create_agent()

    result = agent.kickoff(prompt)

    return str(result)


# =========================================================
# WELCOME SCREEN
# =========================================================

def welcome_screen():

    st.markdown(
        '<div class="hero">'
        '<h1>🎓 Welcome to Study Tutor Agent</h1>'
        '<p>Your personal AI study assistant for learning, practice, quizzes, and study planning.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.subheader("✨ How can you use it?")

    cols = st.columns(4)

    cards = [

        (
            "📚 Learn",
            "Understand concepts with simple explanations and examples."
        ),

        (
            "🧠 Practice",
            "Practice with questions, hints, and guided explanations."
        ),

        (
            "📝 Quiz",
            "Test yourself and learn from your mistakes."
        ),

        (
            "📅 Study Plan",
            "Build a realistic plan around your available time."
        ),
    ]

    for col, (title, description) in zip(
        cols,
        cards
    ):

        with col:

            st.markdown(
                f"""
                <div class="card">
                    <h3>{title}</h3>
                    <p>{description}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    st.subheader("👩‍🎓 Set up your learning session")

    st.write(
        "Tell your tutor a few details so it can personalize your explanations."
    )

    with st.form("welcome_setup_form"):

        name = st.text_input(
            "Your Name",
            placeholder="e.g. Khadija"
        )

        subject = st.selectbox(
            "📚 Subject",
            SUBJECTS,
            index=SUBJECTS.index(
                st.session_state.subject
            ),
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
            ],
        )

        start = st.form_submit_button(
            "🚀 Start Learning",
            use_container_width=True,
            type="primary",
        )

        if start:

            if not name.strip():

                st.error(
                    "Please enter your name."
                )

            elif (
                subject == "➕ Add Your Own Subject"
                and not custom_subject.strip()
            ):

                st.error(
                    "Please enter your own subject."
                )

            elif not topic.strip():

                st.error(
                    "Please enter the topic you want to study."
                )

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
# FIRST OPEN
# =========================================================

if not st.session_state.setup_complete:

    welcome_screen()

    st.stop()


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="hero">'
    '<h1>🎓 Study Tutor Agent</h1>'
    '<p>Your Personal AI Study Assistant</p>'
    '</div>',
    unsafe_allow_html=True,
)


# =========================================================
# TOP NAVIGATION
# =========================================================

nav_cols = st.columns(4)

nav_options = [
    "📚 Learn",
    "🧠 Practice",
    "📝 Quiz",
    "📅 Study Plan"
]

for i, label in enumerate(nav_options):

    with nav_cols[i]:

        if st.button(
            label,
            use_container_width=True,
            key=f"nav_{i}"
        ):

            st.session_state.mode = label.split(
                " ",
                1
            )[1]

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

    st.header("⚙️ Learning Setup")

    current_subject = st.selectbox(
        "📚 Subject",
        SUBJECTS,
        index=SUBJECTS.index(
            st.session_state.subject
        ),
        key="sidebar_subject",
    )

    if current_subject == "➕ Add Your Own Subject":

        new_custom_subject = st.text_input(
            "Your Subject",
            value=st.session_state.custom_subject,
            key="sidebar_custom_subject",
        )

    else:

        new_custom_subject = (
            st.session_state.custom_subject
        )

    new_topic = st.text_input(
        "🎯 Topic",
        value=st.session_state.topic,
        key="sidebar_topic",
    )

    new_level = st.selectbox(
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
        ].index(
            st.session_state.level
        ),
        key="sidebar_level",
    )

    if st.button(
        "💾 Apply Changes",
        use_container_width=True
    ):

        changed = (

            current_subject
            != st.session_state.subject

            or new_custom_subject.strip()
            != st.session_state.custom_subject.strip()

            or new_topic.strip()
            != st.session_state.topic.strip()

            or new_level
            != st.session_state.level
        )

        st.session_state.subject = current_subject

        st.session_state.custom_subject = (
            new_custom_subject.strip()
        )

        st.session_state.topic = (
            new_topic.strip()
        )

        st.session_state.level = new_level

        if changed:

            st.session_state.messages = []

        st.success(
            "Learning setup updated!"
        )

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

        st.success(
            "Conversation cleared!"
        )

        st.rerun()


# =========================================================
# CURRENT SESSION INFO
# =========================================================

st.info(
    f"**Mode:** {st.session_state.mode}  |  "
    f"**Subject:** {get_subject()}  |  "
    f"**Topic:** {st.session_state.topic}"
)


# =========================================================
# SUGGESTED QUESTIONS
# =========================================================

st.subheader("💡 Try asking")

suggestions = {

    "Learn": [

        f"Explain {st.session_state.topic} in simple words with an example.",

        f"What are the important concepts I should know about {st.session_state.topic}?",

        f"Give me a small diagram for {st.session_state.topic}.",
    ],

    "Practice": [

        f"Give me 5 practice questions about {st.session_state.topic}.",

        f"Give me one question about {st.session_state.topic} and wait for my answer.",

        f"Give me a small coding problem related to {st.session_state.topic}.",
    ],

    "Quiz": [

        f"Start a 5-question quiz on {st.session_state.topic}.",

        f"Give me a multiple-choice question about {st.session_state.topic}.",

        "Explain my mistakes after I answer.",
    ],

    "Study Plan": [

        f"Make me a 7-day study plan for {st.session_state.topic}.",

        "Make me a study plan for 2 hours per day.",

        f"How should I revise {st.session_state.topic} before an exam?",
    ],
}


suggestion_cols = st.columns(3)

for i, suggestion in enumerate(
    suggestions.get(
        st.session_state.mode,
        suggestions["Learn"]
    )
):

    with suggestion_cols[i]:

        if st.button(
            suggestion,
            key=f"suggestion_{i}",
            use_container_width=True
        ):

            st.session_state.pending_question = (
                suggestion
            )

            st.rerun()


# =========================================================
# ASK YOUR AI TUTOR
# =========================================================

st.markdown("---")

st.subheader("💬 Ask Your AI Tutor")

st.write(
    f"Ask anything about **{st.session_state.topic}**. "
    "Your AI tutor will explain it according to your level."
)


pending_question = st.session_state.pop(
    "pending_question",
    ""
)


question = st.text_area(
    "Your Question",

    value=pending_question,

    placeholder=(
        "For example: Explain Queue in simple words "
        "with an example and diagram."
    ),

    height=130,

    key="question_box",
)


ask_button = st.button(
    "🚀 Ask Tutor",

    use_container_width=True,

    type="primary",
)


if ask_button:

    if not question.strip():

        st.warning(
            "Please type a question first."
        )

    else:

        clean_question = question.strip()

        st.session_state.messages.append(
            {
                "role": "user",
                "content": clean_question
            }
        )

        with st.spinner(
            "🤖 Study Tutor is thinking..."
        ):

            try:

                answer = ask_tutor(
                    clean_question
                )

            except Exception as error:

                answer = (
                    "Sorry, something went wrong "
                    "while contacting the AI tutor.\n\n"
                    f"**Error:** `{error}`"
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

if st.session_
