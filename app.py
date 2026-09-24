import ast
import operator
import streamlit as st

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Study Tutor Agent",
    page_icon="🎓",
    layout="wide",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .main-header {
        text-align: center;
        padding: 10px 0;
    }

    .main-header h1 {
        margin-bottom: 0;
    }

    .main-header p {
        color: #6b7280;
        margin-top: 5px;
    }

    .mode-box {
        padding: 12px 16px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.25);
        margin: 15px 0;
    }

    .feature-card {
        padding: 18px;
        border-radius: 14px;
        border: 1px solid rgba(128,128,128,0.25);
        min-height: 120px;
    }

    div.stButton > button {
        border-radius: 12px;
        font-weight: 600;
    }

    .footer {
        text-align: center;
        color: #888;
        padding: 25px 0 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CREWAI / GROQ COMPATIBILITY WORKAROUND
# ============================================================

try:
    import crewai.llms.cache as crewai_cache

    crewai_cache.mark_cache_breakpoint = lambda message: message

except Exception:
    pass


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "mode" not in st.session_state:
    st.session_state.mode = "Learn"

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


# ============================================================
# CALCULATOR TOOL
# ============================================================

@tool("Calculator")
def calculator(expression: str) -> str:
    """
    Safely calculate basic mathematical expressions.
    """

    operations = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def evaluate(node):

        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value

            raise ValueError("Only numbers are allowed.")

        if isinstance(node, ast.UnaryOp):

            operation = operations.get(type(node.op))

            if operation is None:
                raise ValueError("Unsupported operator.")

            return operation(evaluate(node.operand))

        if isinstance(node, ast.BinOp):

            operation = operations.get(type(node.op))

            if operation is None:
                raise ValueError("Unsupported operator.")

            left = evaluate(node.left)
            right = evaluate(node.right)

            return operation(left, right)

        raise ValueError("Invalid mathematical expression.")

    try:

        tree = ast.parse(expression, mode="eval")

        result = evaluate(tree.body)

        return str(result)

    except Exception as error:

        return f"Could not calculate: {error}"


# ============================================================
# STUDY PLANNER TOOL
# ============================================================

@tool("Study Planner")
def study_planner(request: str) -> str:
    """
    Creates a framework for a realistic study plan.
    """

    return f"""
Create a realistic study plan for this student request:

{request}

Include:

1. Learning goals
2. Topics
3. Study time
4. Practice
5. Revision
6. Short breaks
7. Daily targets
8. End-of-day review

Keep the workload realistic for a university student.
"""


# ============================================================
# GET SUBJECT
# ============================================================

def get_subject():

    if st.session_state.subject_option == "➕ Add Your Own Subject":

        custom_subject = st.session_state.custom_subject.strip()

        if custom_subject:
            return custom_subject

        return "Custom Subject"

    return st.session_state.subject_option


# ============================================================
# CREATE LLM
# ============================================================

def create_llm():

    api_key = st.secrets.get("GROQ_API_KEY")

    if not api_key:

        raise ValueError(
            "GROQ_API_KEY is missing. "
            "Add it in Streamlit Cloud → Settings → Secrets."
        )

    return LLM(
        model="groq/openai/gpt-oss-120b",
        api_key=api_key,
        temperature=0.3,
        reasoning_effort="medium",
    )


# ============================================================
# CREATE STUDY TUTOR AGENT
# ============================================================

def create_tutor_agent():

    tutor = Agent(
        role="AI Study Tutor",

        goal=(
            "Help university students understand academic subjects, "
            "practice effectively, prepare for quizzes, and create "
            "realistic study plans."
        ),

        backstory=(
            "You are a patient university-level AI study tutor. "
            "You explain difficult concepts in simple English while "
            "remaining academically accurate. You teach instead of "
            "simply giving answers. You use examples, comparisons, "
            "and step-by-step explanations."
        ),

        llm=create_llm(),

        tools=[
            calculator,
            study_planner,
        ],

        allow_delegation=False,

        verbose=False,
    )

    return tutor


# ============================================================
# MODE INSTRUCTIONS
# ============================================================

def get_mode_instructions():

    if st.session_state.mode == "Learn":

        return """
MODE: LEARN

Teach the student.

- Explain the basic idea first.
- Break difficult concepts into smaller parts.
- Give simple examples.
- Give university-level examples when useful.
- Explain why the concept works.
- Use a small text diagram when helpful.
- Use Key Takeaways for difficult topics.
- Do not make the answer unnecessarily long.
"""

    if st.session_state.mode == "Practice":

        return """
MODE: PRACTICE

Help the student practice.

- Give practice questions.
- Do not immediately reveal answers.
- Let the student attempt the questions.
- Check the student's answers carefully.
- Explain mistakes clearly.
- Adjust difficulty to the student's level.
"""

    if st.session_state.mode == "Quiz":

        return """
MODE: QUIZ

Act as a quiz tutor.

- Create objective academic questions.
- Prefer MCQs when the student asks for a quiz.
- Do not reveal answers before the student attempts them.
- Check answers after the student responds.
- Explain mistakes.
- Match the student's selected difficulty.
"""

    return """
MODE: STUDY PLAN

Create a realistic study plan.

- Consider subject and topic.
- Consider available study time when provided.
- Consider number of days when provided.
- Include learning.
- Include practice.
- Include revision.
- Include breaks.
- Divide large topics into manageable sessions.
"""


# ============================================================
# ASK TUTOR
# ============================================================

def ask_tutor(question):

    subject = get_subject()

    student_name = (
        st.session_state.student_name.strip()
        or "Student"
    )

    topic = (
        st.session_state.current_topic.strip()
        or "Not specified"
    )

    level = st.session_state.learning_level

    recent_conversation = ""

    for message in st.session_state.messages[-10:]:

        recent_conversation += (
            f"{message['role'].upper()}: "
            f"{message['content']}\n"
        )

    prompt = f"""
You are a Study Tutor Agent.

STUDENT PROFILE

Student name: {student_name}
Subject: {subject}
Current topic: {topic}
Learning level: {level}
Current mode: {st.session_state.mode}

MODE INSTRUCTIONS

{get_mode_instructions()}

GENERAL TEACHING RULES

1. Answer the student's actual question.
2. Use simple and clear English.
3. Keep the explanation appropriate for a university student.
4. Explain difficult terminology.
5. Use examples when useful.
6. Use headings and bullets when useful.
7. Use the Calculator tool for calculations when appropriate.
8. Use the Study Planner tool for study-plan requests when appropriate.
9. In Practice and Quiz modes, do not reveal answers before the student attempts them.
10. Check submitted answers carefully.
11. Use recent conversation context.
12. Do not claim to remember information that is not present.
13. Avoid unnecessary repetition.
14. Be supportive and teaching-focused.
15. If the question is unclear, make a reasonable interpretation.
16. Do not make responses unnecessarily long.

RECENT CONVERSATION

{recent_conversation}

STUDENT REQUEST

{question}
"""

    tutor = create_tutor_agent()

    task = Task(
        description=prompt,
        expected_output=(
            "A clear, accurate, student-friendly teaching response "
            "appropriate for a university student."
        ),
        agent=tutor,
    )

    crew = Crew(
        agents=[tutor],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )

    result = crew.kickoff()

    return str(result)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-header">
        <h1>🎓 Study Tutor Agent</h1>
        <p>Your Personal AI Study Assistant</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TOP NAVIGATION
# ============================================================

st.markdown("### Choose your learning mode")

col1, col2, col3, col4 = st.columns(4)


with col1:

    if st.button(
        "📚 Learn",
        use_container_width=True,
        type=(
            "primary"
            if st.session_state.mode == "Learn"
            else "secondary"
        ),
    ):

        st.session_state.mode = "Learn"

        st.rerun()


with col2:

    if st.button(
        "🧠 Practice",
        use_container_width=True,
        type=(
            "primary"
            if st.session_state.mode == "Practice"
            else "secondary"
        ),
    ):

        st.session_state.mode = "Practice"

        st.rerun()


with col3:

    if st.button(
        "📝 Quiz",
        use_container_width=True,
        type=(
            "primary"
            if st.session_state.mode == "Quiz"
            else "secondary"
        ),
    ):

        st.session_state.mode = "Quiz"

        st.rerun()


with col4:

    if st.button(
        "📅 Study Plan",
        use_container_width=True,
        type=(
            "primary"
            if st.session_state.mode == "Study Plan"
            else "secondary"
        ),
    ):

        st.session_state.mode = "Study Plan"

        st.rerun()


# ============================================================
# CURRENT MODE
# ============================================================

st.markdown(
    f"""
    <div class="mode-box">
        <strong>Current Mode:</strong>
        {st.session_state.mode}
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("👩‍🎓 Student Profile")

    st.text_input(
        "Your Name",
        placeholder="Enter your name",
        key="student_name",
    )

    st.divider()

    st.subheader("📖 Subject")

    subject_options = [
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

    st.selectbox(
        "Choose Subject",
        subject_options,
        key="subject_option",
    )

    if (
        st.session_state.subject_option
        == "➕ Add Your Own Subject"
    ):

        st.text_input(
            "Enter Your Subject",
            placeholder="e.g. Digital Logic Design",
            key="custom_subject",
        )

        if st.session_state.custom_subject.strip():

            st.success(
                f"Subject added: "
                f"{st.session_state.custom_subject.strip()}"
            )

    st.divider()

    st.subheader("🎯 Current Topic")

    st.text_input(
        "Topic",
        placeholder="e.g. Queue",
        key="current_topic",
    )

    st.divider()

    st.subheader("📊 Learning Level")

    st.selectbox(
        "Choose Level",
        [
            "Beginner",
            "Intermediate",
            "Advanced",
        ],
        key="learning_level",
    )

    st.divider()

    st.caption(
        f"Mode: {st.session_state.mode}"
    )

    st.caption(
        f"Subject: {get_subject()}"
    )

    st.divider()

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.success(
            "Conversation cleared."
        )

        st.rerun()


# ============================================================
# MODE CONTENT
# ============================================================

if st.session_state.mode == "Learn":

    st.subheader("📚 Learn")

    st.write(
        "Ask me anything about your subject "
        "and I will explain it step by step."
    )

    if not st.session_state.messages:

        card1, card2, card3 = st.columns(3)

        with card1:

            st.markdown(
                """
                <div class="feature-card">
                    <h3>💡 Understand</h3>
                    <p>
                    Learn difficult concepts
                    in simple language.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with card2:

            st.markdown(
                """
                <div class="feature-card">
                    <h3>🧩 Examples</h3>
                    <p>
                    Learn through practical
                    and academic examples.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with card3:

            st.markdown(
                """
                <div class="feature-card">
                    <h3>🎯 Improve</h3>
                    <p>
                    Ask follow-up questions
                    until the concept is clear.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )


elif st.session_state.mode == "Practice":

    st.subheader("🧠 Practice")

    st.write(
        "Practice your selected topic "
        "and receive feedback."
    )

    st.info(
        f"Subject: {get_subject()} | "
        f"Topic: "
        f"{st.session_state.current_topic or 'Not specified'} | "
        f"Level: {st.session_state.learning_level}"
    )


elif st.session_state.mode == "Quiz":

    st.subheader("📝 Quiz")

    st.write(
        "Ask the Study Tutor to create "
        "a quiz for your selected subject."
    )

    st.info(
        f"Subject: {get_subject()} | "
        f"Topic: "
        f"{st.session_state.current_topic or 'General'} | "
        f"Level: {st.session_state.learning_level}"
    )

    st.caption(
        "Example: Give me 5 MCQs on queues without answers."
    )


elif st.session_state.mode == "Study Plan":

    st.subheader("📅 Study Plan")

    st.write(
        "Tell the tutor what you want to study "
        "and how much time you have."
    )

    st.info(
        f"Subject: {get_subject()} | "
        f"Topic: "
        f"{st.session_state.current_topic or 'Not specified'}"
    )

    st.caption(
        "Example: Make me a 7-day Data Structures "
        "plan for 2 hours per day."
    )


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

if st.session_state.messages:

    st.markdown("---")

    st.subheader("💬 Tutor Conversation")

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(
                message["content"]
            )


# ============================================================
# CHAT INPUT
# ============================================================

placeholders = {
    "Learn": "Ask me something you want to learn...",
    "Practice": (
        "Ask for practice questions "
        "or submit your answer..."
    ),
    "Quiz": "Ask me to create a quiz...",
    "Study Plan": (
        "Tell me what you want to study "
        "and your available time..."
    ),
}

prompt = st.chat_input(
    placeholders[st.session_state.mode]
)


# ============================================================
# PROCESS USER MESSAGE
# ============================================================

if prompt:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user"):

        st.markdown(prompt)

    with st.chat_message("assistant"):

        with st.spinner(
            "🤖 Study Tutor is thinking..."
        ):

            try:

                response = ask_tutor(prompt)

                st.markdown(response)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response,
                    }
                )

            except Exception as error:

                st.error(
                    "Something went wrong. "
                    "Please check your Groq API key "
                    "and deployment settings."
                )

                st.code(
                    str(error),
                    language="text",
                )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Study Tutor Agent • CrewAI • Groq • GPT-OSS 120B
    </div>
    """,
    unsafe_allow_html=True,
    )
