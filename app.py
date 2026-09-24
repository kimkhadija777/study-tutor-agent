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
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main page */
    .main {
        padding-top: 1rem;
    }

    /* Header */
    .main-header {
        text-align: center;
        padding: 10px 0 4px 0;
    }

    .main-header h1 {
        margin-bottom: 0;
        font-size: 2.3rem;
    }

    .main-header p {
        color: #6b7280;
        font-size: 1rem;
        margin-top: 4px;
    }

    /* Mode buttons */
    div.stButton > button {
        border-radius: 12px;
        min-height: 48px;
        font-weight: 600;
    }

    /* Cards */
    .feature-card {
        padding: 18px;
        border-radius: 14px;
        border: 1px solid rgba(128,128,128,0.25);
        background-color: rgba(128,128,128,0.05);
        min-height: 120px;
    }

    .feature-card h3 {
        margin-top: 0;
    }

    /* Current mode */
    .mode-banner {
        padding: 12px 16px;
        border-radius: 12px;
        margin: 15px 0;
        border: 1px solid rgba(128,128,128,0.25);
        background-color: rgba(128,128,128,0.06);
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #888;
        padding: 25px 0 10px 0;
        font-size: 0.85rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CREWAI CACHE COMPATIBILITY FIX
# ============================================================

# Some CrewAI/LiteLLM + Groq combinations may add a
# cache_breakpoint field that Groq does not accept.
# This keeps the workaround from the previous working version.

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

if "custom_subject" not in st.session_state:
    st.session_state.custom_subject = ""

if "quiz_started" not in st.session_state:
    st.session_state.quiz_started = False

if "study_plan_started" not in st.session_state:
    st.session_state.study_plan_started = False


# ============================================================
# TOOLS
# ============================================================

@tool("Calculator")
def calculator(expression: str) -> str:
    """
    Safely calculate a mathematical expression.

    Supports basic arithmetic:
    +, -, *, /, //, %, **
    and parentheses.
    """

    allowed_operators = {
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
            operation = allowed_operators.get(type(node.op))

            if operation is None:
                raise ValueError("Unsupported operator.")

            return operation(evaluate(node.operand))

        if isinstance(node, ast.BinOp):
            operation = allowed_operators.get(type(node.op))

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
        return f"Could not calculate the expression: {error}"


@tool("Study Planner")
def study_planner(request: str) -> str:
    """
    Creates a study-plan framework based on the student's request.
    """

    return f"""
Create a realistic study plan for the following request:

{request}

The plan should include:

1. Main learning goal
2. Topics to study
3. Suggested time blocks
4. Practice time
5. Revision time
6. Short breaks
7. End-of-day review
8. A small achievable target for each study session

Keep the plan realistic and student-friendly.
"""


# ============================================================
# LLM
# ============================================================

def create_llm():

    api_key = st.secrets.get("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. Add it in Streamlit Cloud "
            "under Settings → Secrets."
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

    llm = create_llm()

    tutor = Agent(
        role="AI Study Tutor",
        goal=(
            "Help university students understand academic subjects "
            "clearly, practice effectively, prepare for quizzes, "
            "and create realistic study plans."
        ),
        backstory=(
            "You are a patient university-level AI study tutor. "
            "You explain difficult concepts in simple English while "
            "keeping explanations academically accurate. "
            "You teach instead of simply giving answers. "
            "You use examples, step-by-step reasoning, comparisons, "
            "practice questions, and summaries when useful."
        ),
        llm=llm,
        tools=[calculator, study_planner],
        allow_delegation=False,
        verbose=False,
    )

    return tutor


# ============================================================
# GET CURRENT SUBJECT
# ============================================================

def get_subject():

    subject_option = st.session_state.get("subject_option", "Computer Science")

    if subject_option == "➕ Add Your Own Subject":

        custom_subject = st.session_state.get(
            "custom_subject",
            ""
        ).strip()

        if custom_subject:
            return custom_subject

        return "Custom Subject"

    return subject_option


# ============================================================
# MODE INSTRUCTIONS
# ============================================================

def get_mode_instructions():

    mode = st.session_state.mode

    if mode == "Learn":

        return """
MODE: LEARN

Teach the student.

Requirements:
- Explain the concept clearly.
- Start with the basic idea.
- Break difficult concepts into smaller parts.
- Give a simple example.
- Give a university-level example when useful.
- Use a small diagram or visual representation when helpful.
- Explain WHY the concept works.
- End with Key Takeaways when the topic is difficult.
- Do not make the answer unnecessarily long.
"""

    if mode == "Practice":

        return """
MODE: PRACTICE

Help the student practice.

Requirements:
- Give practice questions based on the current subject and topic.
- Do not immediately reveal answers unless the student asks.
- Ask the student to attempt the questions.
- When the student gives an answer, evaluate it.
- Explain mistakes clearly.
- Encourage understanding rather than memorization.
- Adjust difficulty based on the student's level.
"""

    if mode == "Quiz":

        return """
MODE: QUIZ

Act as a quiz tutor.

Requirements:
- Create objective academic questions.
- Prefer MCQs unless the student requests another format.
- Do not reveal answers before the student attempts them.
- After the student answers, explain which answers are correct.
- Explain mistakes.
- Keep questions relevant to the selected subject and topic.
- Match the selected difficulty level.
- If the student asks for a new quiz, create fresh questions.
"""

    if mode == "Study Plan":

        return """
MODE: STUDY PLAN

Act as a study-planning assistant.

Requirements:
- Create a realistic study plan.
- Consider the student's subject, topic, level, available time,
  and number of days when provided.
- Include learning, practice, revision, and breaks.
- Avoid unrealistic workloads.
- Divide large topics into manageable sessions.
- Use the Study Planner tool when appropriate.
"""

    return ""


# ============================================================
# ASK THE TUTOR
# ============================================================

def ask_tutor(question):

    subject = get_subject()

    student_name = st.session_state.get(
        "student_name",
        "Student"
    )

    level = st.session_state.get(
        "learning_level",
        "Beginner"
    )

    topic = st.session_state.get(
        "current_topic",
        ""
    )

    recent_messages = st.session_state.messages[-10:]

    conversation_context = ""

    for message in recent_messages:
        conversation_context += (
            f"{message['role'].upper()}: "
            f"{message['content']}\n"
        )

    mode_instructions = get_mode_instructions()

    prompt = f"""
You are working as a Study Tutor Agent.

STUDENT PROFILE
Student name: {student_name}
Subject: {subject}
Learning level: {level}
Current topic: {topic if topic else "Not specified"}
Current mode: {st.session_state.mode}

{mode_instructions}

IMPORTANT TEACHING RULES

1. Answer the student's actual question.
2. Use simple, clear English.
3. Keep the explanation appropriate for a university student.
4. Do not assume the student already understands advanced concepts.
5. Explain difficult terminology.
6. Use examples when helpful.
7. Use bullet points and headings when they improve clarity.
8. If mathematics is involved, use the Calculator tool when useful.
9. If the student asks for a study plan, use the Study Planner tool when useful.
10. If the student asks for practice, do not automatically reveal answers.
11. If the student provides an answer, check it carefully.
12. Use the recent conversation to maintain context.
13. Do not claim to remember information that is not present in the conversation.
14. Avoid unnecessary repetition.
15. Be supportive and encouraging.
16. Focus on teaching the student how to understand the concept.
17. If the question is unclear, make a reasonable interpretation and explain it.
18. Do not make the response unnecessarily long.

RECENT CONVERSATION

{conversation_context}

STUDENT'S NEW REQUEST

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
# TOP HEADER
# ============================================================

st.markdown(
    """
    <div class="main-header">
        <h1>🎓 Study Tutor Agent</h1>
        <p>Your personal AI study assistant</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TOP NAVIGATION
# ============================================================

st.markdown("### Choose your learning mode")

nav1, nav2, nav3, nav4 = st.columns(4)

with nav1:
    if st.button(
        "📚 Learn",
        use_container_width=True,
        type="primary" if st.session_state.mode == "Learn" else "secondary",
    ):
        st.session_state.mode = "Learn"
        st.rerun()

with nav2:
    if st.button(
        "🧠 Practice",
        use_container_width=True,
        type="primary" if st.session_state.mode == "Practice" else "secondary",
    ):
        st.session_state.mode = "Practice"
        st.rerun()

with nav3:
    if st.button(
        "📝 Quiz",
        use_container_width=True,
        type="primary" if st.session_state.mode == "Quiz" else "secondary",
    ):
        st.session_state.mode = "Quiz"
        st.session_state.quiz_started = True
        st.rerun()

with nav4:
    if st.button(
        "📅 Study Plan",
        use_container_width=True,
        type="primary" if st.session_state.mode == "Study Plan" else "secondary",
    ):
        st.session_state.mode = "Study Plan"
        st.session_state.study_plan_started = True
        st.rerun()


# ============================================================
# CURRENT MODE
# ============================================================

st.markdown(
    f"""
    <div class="mode-banner">
        <strong>Current Mode:</strong> {st.session_state.mode}
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
        value="",
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

    if st.session_state.subject_option == "➕ Add Your Own Subject":

        st.text_input(
            "Enter Your Subject",
            placeholder="e.g. Digital Logic Design",
            key="custom_subject",
        )

        if st.session_state.custom_subject.strip():
            st.success(
                f"Subject added: {st.session_state.custom_subject}"
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
        st.session_state.quiz_started = False
        st.session_state.study_plan_started = False

        st.success("Conversation cleared.")

        st.rerun()


# ============================================================
# MAIN CONTENT — MODE INTRODUCTION
# ============================================================

if st.session_state.mode == "Learn":

    st.subheader("📚 Learn")

    st.write(
        "Ask me anything about your subject. "
        "I'll explain it step by step."
    )

    if not st.session_state.messages:

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                """
                <div class="feature-card">
                    <h3>💡 Understand</h3>
                    <p>Learn difficult concepts in simple language.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                """
                <div class="feature-card">
                    <h3>🧩 Examples</h3>
                    <p>Learn through practical and academic examples.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown(
                """
                <div class="feature-card">
                    <h3>🎯 Improve</h3>
                    <p>Ask follow-up questions until the concept is clear.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


elif st.session_state.mode == "Practice":

    st.subheader("🧠 Practice")

    st.write(
        "Practice your selected topic and get feedback on your answers."
    )

    st.info(
        f"Subject: {get_subject()}  |  "
        f"Topic: {st.session_state.current_topic or 'Not specified'}  |  "
        f"Level: {st.session_state.learning_level}"
    )


elif st.session_state.mode == "Quiz":

    st.subheader("📝 Quiz")

    st.write(
        "Ask the Study Tutor to create a quiz for your selected subject."
    )

    st.info(
        f"Quiz subject: {get_subject()}  |  "
        f"Topic: {st.session_state.current_topic or 'General'}  |  "
        f"Level: {st.session_state.learning_level}"
    )

    st.caption(
        "Example: 'Give me 5 MCQs on queues without answers.'"
    )


elif st.session_state.mode == "Study Plan":

    st.subheader("📅 Study Plan")

    st.write(
        "Tell the tutor what you need to study and how much time you have."
    )

    st.info(
        f"Subject: {get_subject()}  |  "
        f"Topic: {st.session_state.current_topic or 'Not specified'}"
    )

    st.caption(
        "Example: 'Make me a 7-day Data Structures plan "
        "for 2 hours per day.'"
    )


# ============================================================
# CHAT HISTORY
# ============================================================

if st.session_state.messages:

    st.markdown("---")
    st.subheader("💬 Tutor Conversation")

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])


# ============================================================
# CHAT INPUT
# ============================================================

prompt_placeholder = {
    "Learn": "Ask me something you want to learn...",
    "Practice": "Ask for practice questions or submit your answer...",
    "Quiz": "Ask me to create a quiz...",
    "Study Plan": "Tell me what you want to study and your available time...",
}

prompt = st.chat_input(
    prompt_placeholder[st.session_state.mode]
)


# ============================================================
# HANDLE USER MESSAGE
# ============================================================

if prompt:

    # Save user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate tutor response
    with st.chat_message("assistant"):

        with st.spinner("🤖 Study Tutor is thinking..."):

            try:

                response = ask_tutor(prompt)

                st.markdown(response)

                st.session_state.messages.append(
                    {
                  
