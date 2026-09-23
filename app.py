import ast
import operator as op
import streamlit as st

from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Study Tutor Agent",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# CUSTOM TOOLS
# ============================================================

@tool("Calculator")
def calculator(expression: str) -> str:
    """
    Safely calculate basic mathematical expressions.
    Supports +, -, *, /, %, ** and parentheses.
    """

    allowed_operators = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.Mod: op.mod,
        ast.Pow: op.pow,
        ast.USub: op.neg,
        ast.UAdd: op.pos,
    }

    def evaluate(node):

        if isinstance(node, ast.Expression):
            return evaluate(node.body)

        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Only numbers are allowed.")

        if isinstance(node, ast.BinOp):
            operator_type = type(node.op)

            if operator_type not in allowed_operators:
                raise ValueError("Operator not allowed.")

            left = evaluate(node.left)
            right = evaluate(node.right)

            return allowed_operators[operator_type](left, right)

        if isinstance(node, ast.UnaryOp):
            operator_type = type(node.op)

            if operator_type not in allowed_operators:
                raise ValueError("Operator not allowed.")

            return allowed_operators[operator_type](
                evaluate(node.operand)
            )

        raise ValueError("Invalid mathematical expression.")

    try:
        tree = ast.parse(expression, mode="eval")
        result = evaluate(tree)

        return f"Calculation result: {result}"

    except Exception:
        return (
            "I could not calculate that expression. "
            "Please use a simple mathematical expression."
        )


@tool("Study Planner")
def study_planner(topic: str) -> str:
    """
    Creates a simple study-plan structure for a topic.
    """

    return f"""
Create a study plan for the topic: {topic}

The plan should contain:
1. Learn the basic concept
2. Understand an example
3. Practice
4. Review mistakes
5. Quick revision
"""


# ============================================================
# MEMORY
# ============================================================

def initialize_memory():

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "student_name" not in st.session_state:
        st.session_state.student_name = ""

    if "subject" not in st.session_state:
        st.session_state.subject = "Computer Science"

    if "level" not in st.session_state:
        st.session_state.level = "Beginner"

    if "topic" not in st.session_state:
        st.session_state.topic = ""


def clear_memory():

    st.session_state.messages = []


def get_conversation_memory():

    messages = st.session_state.messages

    if not messages:
        return "No previous conversation."

    recent_messages = messages[-10:]

    conversation = []

    for message in recent_messages:

        role = message["role"].upper()
        content = message["content"]

        conversation.append(
            f"{role}: {content}"
        )

    return "\n".join(conversation)


# ============================================================
# CREATE STUDY TUTOR AGENT
# ============================================================

def create_tutor_agent():

    groq_api_key = st.secrets["GROQ_API_KEY"]

    llm = LLM(
        model="groq/openai/gpt-oss-120b",
        api_key=groq_api_key,
        temperature=0.3,
        reasoning_effort="medium"
    )

    tutor = Agent(

        role="University Study Tutor",

        goal=(
            "Help students understand academic concepts clearly, "
            "practice what they learn, identify misunderstandings, "
            "and become more confident learners."
        ),

        backstory=(
            "You are a patient and friendly university study tutor. "
            "You explain difficult concepts in simple language while "
            "maintaining university-level accuracy. "
            "You use examples, step-by-step explanations, "
            "practice questions, and constructive feedback. "
            "You never intentionally overwhelm the student."
        ),

        llm=llm,

        tools=[
            calculator,
            study_planner
        ],

        allow_delegation=False,

        verbose=False
    )

    return tutor


# ============================================================
# RUN TUTOR
# ============================================================

def ask_tutor(question):

    tutor = create_tutor_agent()

    conversation = get_conversation_memory()

    context = f"""
Student information:

Name:
{st.session_state.student_name or "Not provided"}

Subject:
{st.session_state.subject}

Current topic:
{st.session_state.topic or "Not provided"}

Learning level:
{st.session_state.level}


Recent conversation:

{conversation}
"""

    task = Task(

        description=f"""
You are helping a student learn.

{context}

Current student request:

{question}


Follow these rules:

1. Answer the student's actual question.
2. Use simple and clear language.
3. Keep the explanation appropriate for a university student.
4. Break difficult concepts into smaller parts.
5. Give an example when useful.
6. If a calculation is required, use the Calculator tool.
7. If the student asks for a study plan, use the Study Planner tool.
8. If the student asks for practice questions, create useful
   questions and do not immediately reveal answers unless requested.
9. If the student gives an answer, evaluate it and explain mistakes.
10. Remember the recent conversation when answering follow-up questions.
11. Do not claim that you remember information that is not present
    in the provided conversation.
12. Be encouraging but focus on teaching rather than simply giving
    an answer.
""",

        expected_output=(
            "A clear, accurate, student-friendly tutoring response."
        ),

        agent=tutor
    )

    crew = Crew(

        agents=[tutor],

        tasks=[task],

        verbose=False
    )

    result = crew.kickoff()

    return str(result)


# ============================================================
# INITIALIZE MEMORY
# ============================================================

initialize_memory()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🎓 Study Tutor")

    st.markdown(
        "Personal AI tutor for learning and practice."
    )

    st.divider()

    st.subheader("👩‍🎓 Student")

    st.session_state.student_name = st.text_input(
        "Your name",
        value=st.session_state.student_name,
        placeholder="Enter your name"
    )

    st.subheader("📚 Study Settings")

    st.session_state.subject = st.selectbox(
        "Subject",
        [
            "Computer Science",
            "Data Structures",
            "Artificial Intelligence",
            "Generative AI",
            "Cybersecurity",
            "Computer Networks",
            "Software Engineering",
            "Database Systems",
            "Other"
        ]
    )

    st.session_state.level = st.selectbox(
        "Learning level",
        [
            "Beginner",
            "Intermediate",
            "Advanced"
        ]
    )

    st.session_state.topic = st.text_input(
        "Current topic",
        value=st.session_state.topic,
        placeholder="e.g. Stack"
    )

    st.divider()

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True
    ):

        clear_memory()
        st.rerun()

    st.divider()

    st.caption(
        "Powered by CrewAI + Groq GPT-OSS 120B"
    )


# ============================================================
# MAIN PAGE
# ============================================================

st.title("🎓 Study Tutor Agent")

st.markdown(
    "Learn concepts, practice questions, solve problems, "
    "and study step by step with your AI tutor."
)


# ============================================================
# WELCOME MESSAGE
# ============================================================

if not st.session_state.messages:

    st.info(
        "👋 Welcome! Tell me what you want to learn. "
        "For example: 'Explain stacks with a simple example.'"
    )


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask your Study Tutor..."
)


if question:

    # Save user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    # Display user message
    with st.chat_message("user"):

        st.markdown(question)


    # Generate tutor response
    with st.chat_message("assistant"):

        with st.spinner(
            "🧠 Your tutor is thinking..."
        ):

            try:

                answer = ask_tutor(question)

                st.markdown(answer)

            except Exception as error:

                answer = (
                    "⚠️ I couldn't process that request.\n\n"
                    "Please check that your Groq API key is correctly "
                    "configured in Streamlit Secrets.\n\n"
                    f"Technical error: `{error}`"
                )

                st.error(answer)


    # Save assistant message
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
      )
