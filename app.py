import ast
import operator as op

import streamlit as st

# ---------------------------------------------------------
# IMPORTANT:
# CrewAI currently has a cache_breakpoint issue with
# non-Anthropic providers such as Groq.
#
# This disables the problematic cache marker before
# CrewAI sends messages to Groq.
# ---------------------------------------------------------

try:
    import crewai.llms.cache as crewai_cache

    crewai_cache.mark_cache_breakpoint = lambda message: message

except Exception:
    pass


from crewai import Agent, Crew, Task, LLM
from crewai.tools import tool


# =========================================================
# STREAMLIT PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Study Tutor Agent",
    page_icon="🎓",
    layout="wide"
)


# =========================================================
# CUSTOM TOOL 1: CALCULATOR
# =========================================================

@tool("Calculator")
def calculator(expression: str) -> str:
    """
    Safely calculate a basic mathematical expression.

    Examples:
    10 + 5
    20 * 4
    (100 / 5) + 7
    """

    operators = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.Mod: op.mod,
        ast.Pow: op.pow,
        ast.USub: op.neg,
        ast.UAdd: op.pos,
    }

    def calculate(node):

        if isinstance(node, ast.Expression):
            return calculate(node.body)

        if isinstance(node, ast.Constant):

            if isinstance(node.value, (int, float)):
                return node.value

            raise ValueError("Only numbers are allowed.")

        if isinstance(node, ast.BinOp):

            operator_type = type(node.op)

            if operator_type not in operators:
                raise ValueError("Operator is not allowed.")

            left = calculate(node.left)
            right = calculate(node.right)

            return operators[operator_type](left, right)

        if isinstance(node, ast.UnaryOp):

            operator_type = type(node.op)

            if operator_type not in operators:
                raise ValueError("Operator is not allowed.")

            return operators[operator_type](
                calculate(node.operand)
            )

        raise ValueError("Invalid expression.")

    try:

        tree = ast.parse(
            expression,
            mode="eval"
        )

        result = calculate(tree)

        return f"Calculation result: {result}"

    except Exception:

        return (
            "I could not calculate that expression. "
            "Please provide a simple mathematical expression."
        )


# =========================================================
# CUSTOM TOOL 2: STUDY PLANNER
# =========================================================

@tool("Study Planner")
def study_planner(topic: str) -> str:
    """
    Creates a simple study-plan structure for a topic.
    """

    return f"""
Study topic: {topic}

Suggested learning sequence:

1. Understand the basic concept
2. Learn the important terminology
3. Study a simple example
4. Practice with questions
5. Review mistakes
6. Revise the key points
"""


# =========================================================
# MEMORY
# =========================================================

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


def get_conversation_memory():

    if not st.session_state.messages:

        return "There is no previous conversation."


    # Keep only the most recent messages.
    recent_messages = st.session_state.messages[-10:]

    conversation = []

    for message in recent_messages:

        role = message["role"].upper()
        content = message["content"]

        conversation.append(
            f"{role}: {content}"
        )

    return "\n".join(conversation)


def clear_memory():

    st.session_state.messages = []


# =========================================================
# CREATE LLM
# =========================================================

def create_llm():

    api_key = st.secrets["GROQ_API_KEY"]

    return LLM(
        model="groq/openai/gpt-oss-120b",
        api_key=api_key,
        temperature=0.3,
        reasoning_effort="medium"
    )


# =========================================================
# CREATE STUDY TUTOR AGENT
# =========================================================

def create_tutor_agent():

    llm = create_llm()

    tutor = Agent(

        role="University Study Tutor",

        goal=(
            "Help students understand academic concepts clearly, "
            "practice effectively, solve problems, and improve "
            "their understanding."
        ),

        backstory=(
            "You are a friendly and patient university study tutor. "
            "You teach difficult concepts in simple English while "
            "maintaining academic accuracy. "
            "You use examples, step-by-step explanations, "
            "practice questions, and constructive feedback. "
            "You adapt explanations to the student's level."
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


# =========================================================
# RUN STUDY TUTOR
# =========================================================

def ask_tutor(question):

    tutor = create_tutor_agent()

    conversation = get_conversation_memory()

    student_name = (
        st.session_state.student_name
        if st.session_state.student_name
        else "Not provided"
    )

    subject = st.session_state.subject

    level = st.session_state.level

    topic = (
        st.session_state.topic
        if st.session_state.topic
        else "Not provided"
    )


    task_description = f"""
You are tutoring a university student.

STUDENT INFORMATION
-------------------

Name:
{student_name}

Subject:
{subject}

Learning level:
{level}

Current topic:
{topic}


RECENT CONVERSATION
-------------------

{conversation}


CURRENT STUDENT REQUEST
-----------------------

{question}


TEACHING RULES
--------------

1. Answer the student's actual question.

2. Use simple and clear English.

3. Keep the explanation appropriate for a
   university student.

4. Break difficult concepts into smaller parts.

5. Give examples when useful.

6. If mathematics or arithmetic is required,
   use the Calculator tool.

7. If the student requests a study plan,
   use the Study Planner tool.

8. If the student asks for practice questions,
   create useful questions.

9. If the student gives an answer,
   check it carefully and explain mistakes.

10. Use the recent conversation to understand
    follow-up questions.

11. Do not claim to remember information that
    is not present in the provided conversation.

12. Do not unnecessarily repeat the same
    explanation.

13. Be encouraging, but focus on teaching.

14. Use headings, bullet points, and examples
    when they improve readability.

15. For difficult topics, finish with a short
    "Key Takeaways" section.

16. If the student's question is unclear,
    make a reasonable interpretation and
    explain what you understood.
"""


    task = Task(

        description=task_description,

        expected_output=(
            "A clear, accurate, beginner-friendly "
            "university-level tutoring response."
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


# =========================================================
# INITIALIZE APP MEMORY
# =========================================================

initialize_memory()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("🎓 Study Tutor")

    st.caption(
        "Your personal AI learning assistant"
    )

    st.divider()


    # -----------------------------------------------------
    # STUDENT INFORMATION
    # -----------------------------------------------------

    st.subheader("👩‍🎓 Student")

    st.session_state.student_name = st.text_input(
        "Your name",
        value=st.session_state.student_name,
        placeholder="Enter your name"
    )


    # -----------------------------------------------------
    # STUDY SETTINGS
    # -----------------------------------------------------

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
            "HTML & Web Development",
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
        placeholder="Example: Stack"
    )


    st.divider()


    # -----------------------------------------------------
    # CLEAR CHAT
    # -----------------------------------------------------

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True
    ):

        clear_memory()

        st.rerun()


    st.divider()

    st.caption(
        "CrewAI • Groq • GPT-OSS 120B"
    )


# =========================================================
# MAIN HEADER
# =========================================================

st.title("🎓 Study Tutor Agent")

st.markdown(
    """
Learn concepts, practice questions, solve problems,
and study step by step with your AI tutor.
"""
)


# =========================================================
# QUICK START
# =========================================================

if not st.session_state.messages:

    st.info(
        "👋 Welcome! Start by asking me about any topic "
        "you want to learn."
    )


    st.markdown("### 💡 Try one of these")

    col1, col2, col3 = st.columns(3)


    with col1:

        st.markdown(
            """
            **📚 Learn**

            Explain Stack in Data Structures
            with a simple example.
            """
        )


    with col2:

        st.markdown(
            """
            **🧠 Practice**

            Give me 5 MCQs about queues
            without showing the answers.
            """
        )


    with col3:

        st.markdown(
            """
            **📅 Plan**

            Create a 7-day study plan
            for learning Data Structures.
            """
        )


# =========================================================
# DISPLAY CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# =========================================================
# USER INPUT
# =========================================================

question = st.chat_input(
    "Ask your Study Tutor..."
)


if question:

    # -----------------------------------------------------
    # SAVE USER MESSAGE
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )


    # -----------------------------------------------------
    # DISPLAY USER MESSAGE
    # -----------------------------------------------------

    with st.chat_message("user"):

        st.markdown(question)


    # -----------------------------------------------------
    # GENERATE AI RESPONSE
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "🧠 Your tutor is thinking..."
        ):

            try:

                answer = ask_tutor(question)

                st.markdown(answer)


            except Exception as error:

                st.error(
                    "Something went wrong while contacting "
                    "the Study Tutor."
                )

                st.caption(
                    "Please check your Streamlit Secret and "
                    "the application logs."
                )

                st.code(
                    str(error)
                )

                answer = (
                    "I couldn't generate a response because "
                    "the AI service returned an error."
                )


    # -----------------------------------------------------
    # SAVE AI MESSAGE
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
            )
