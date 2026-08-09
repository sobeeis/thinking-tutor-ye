import streamlit as st
import json

from dotenv import load_dotenv
from langchain_upstage import ChatUpstage
from langchain_core.messages import HumanMessage, AIMessage


load_dotenv()


st.set_page_config(
    page_title="Thinking feedback Tutor",
    page_icon="🧠",
)

#제목
st.title("🧠 Thinking feedback Tutor")

#설명
st.caption(
    "답을 대신하는 AI가 아니라\n"
    "학생의 능동적인 생각을 도와주는 해룡고등학교 특제 AI, made by 예찬"
)

llm = ChatUpstage()

def chatbot(messages):
    chat_history = []

    for message in messages:
        if message["role"] == "user":
            chat_history.append(
                HumanMessage(content=message["content"])
            )

        elif message["role"] == "assistant":
            chat_history.append(
                AIMessage(content=message["content"])
            )

    response = llm.invoke(chat_history)

    return response.content

#답변 평가 함수
def evaluate_answer(
    question,
    reference_answer,
    student_answer
):
    
    prompt = f"""
너는 학생의 개념 이해도를 평가하는 교육용 평가자다.

질문:
{question}

평가 기준 답안:
{reference_answer}

학생 답변:
{student_answer}

반드시 위의 동일한 기준 답안을 바탕으로 평가하라.
...


1. semantic_score
- 기준 답안과 의미적으로 얼마나 일치하는가
- 0~100점

2. concept_score
- 핵심 개념을 얼마나 정확하게 포함했는가
- 0~100점

3. reasoning_score
- 단순히 결론만 말한 것이 아니라
  이유와 인과관계를 논리적으로 설명했는가
- 0~100점

4. misconception
다음 중 하나:
- none
- minor
- major
- critical

5. feedback
학생이 잘 이해한 점과 부족한 점을 짧게 설명하라.

반드시 아래 JSON 형식으로만 답하라.

{{
    "semantic_score": 0,
    "concept_score": 0,
    "reasoning_score": 0,
    "misconception": "none",
    "feedback": ""
}}
"""

    response = llm.invoke(prompt)

    evaluation = json.loads(response.content)

    return evaluation

def calculate_score(evaluation):

    semantic = evaluation["semantic_score"]
    concept = evaluation["concept_score"]
    reasoning = evaluation["reasoning_score"]

    misconception = evaluation["misconception"]

    score = (
        semantic * 0.25
        + concept * 0.45
        + reasoning * 0.30
    )

    penalty = {
        "none": 0,
        "minor": 10,
        "major": 25,
        "critical": 40
    }

    score -= penalty.get(misconception, 0)

    score = max(0, min(100, score))

    return round(score, 1)

def get_level(score):

    if score < 40:
        return 1

    elif score < 60:
        return 2

    elif score < 80:
        return 3

    elif score < 90:
        return 4

    else:
        return 5

def get_feedback_instruction(level):

    if level == 1:
        return """
학생이 개념을 거의 이해하지 못한 상태다.
정답을 바로 길게 설명하지 말고,
아주 쉬운 예시와 핵심 개념 하나만 제시한 뒤
짧은 질문을 하나 하라.
"""

    elif level == 2:
        return """
학생이 일부 개념은 이해했지만 중요한 연결이 부족하다.
맞게 이해한 부분을 먼저 알려주고,
빠진 핵심 개념에 대한 힌트를 주어라
"""

    elif level == 3:
        return """
학생이 기본 개념은 이해했다.
부족한 인과관계나 근거를 짚고,
왜 그런지 설명하게 하는 질문을 하나 하라.
"""

    elif level == 4:
        return """
학생이 대부분 정확히 이해했다.
표현을 더 정교하게 다듬게 해라.
"""


def generate_feedback(question, student_answer, evaluation, level):

    instruction = get_feedback_instruction(level)

    prompt = f"""
너는 학생의 사고를 돕는 피드백 튜터다.

원래 질문:
{question}

학생 답변:
{student_answer}

평가 결과:
- 의미 일치도: {evaluation["semantic_score"]}
- 핵심개념 충족도: {evaluation["concept_score"]}
- 추론·설명력: {evaluation["reasoning_score"]}
- 오개념: {evaluation["misconception"]}
- 평가 의견: {evaluation["feedback"]}

현재 이해 수준:
Level {level}

피드백 원칙:
{instruction}

반드시 다음 형식으로 답하라.

잘 이해한 점:
...

보완할 점:
...

정답 전체를 대신 작성하지 마라.
학생이 자신의 기존 답변을 다시 수정할 수 있을 정도의 피드백만 제공하라.

"""

    response = llm.invoke(prompt)

    return response.content

#다음 질문 생성기
def generate_next_question(
    question,
    student_answer,
    evaluation,
    level
):

    prompt = f"""
너는 학생의 사고를 확장하는 튜터다.

원래 질문:
{question}

학생의 수정된 답변:
{student_answer}

현재 이해 수준:
Level {level}

평가:
{evaluation["feedback"]}

학생이 이미 이해한 내용을 반복해서 묻지 마라.

현재 이해 수준에 맞게
한 단계 더 깊게 생각할 수 있는 질문을
딱 하나만 만들어라.

질문은 짧고 명확하게 작성하라.
정답은 알려주지 마라.
"""

    response = llm.invoke(prompt)

    return response.content

#기준답안 한개 생성하는 함수
def generate_reference_answer(question):

    prompt = f"""
다음 질문에 대한 정확하고 핵심적인 기준 답안을 작성하라.

질문:
{question}

학생 답변 평가에 사용할 것이므로
핵심 개념과 인과관계를 빠짐없이 포함하되
불필요하게 길게 쓰지 마라.

답안만 작성하라.
"""

    response = llm.invoke(prompt)

    return response.content

#전이 질문 코멘트기
def generate_transfer_comment(
    original_question,
    transfer_question,
    student_answer
):

    prompt = f"""
너는 학생의 사고를 돕는 학습 튜터다.

학생은 앞에서 하나의 개념을 학습했고,
이제 그 개념을 새로운 상황에 적용하는 질문에 답했다.

원래 학습 질문:
{original_question}

전이 질문:
{transfer_question}

학생 답변:
{student_answer}

학생의 답변을 평가하되 점수나 Level은 제시하지 마라.

다음 원칙을 지켜라.

1. 학생이 배운 개념을 새로운 상황에 어떻게 적용했는지 확인하라.
2. 잘 적용한 부분이 있다면 구체적으로 알려줘라.
3. 중요한 오류나 부족한 점이 있다면 짧게 교정하라.
4. 깔끔하고 정확하게 개념을 전달하라.
5. 마지막에는 이번 학습에서 학생이 이해한 핵심을 한 문장으로 정리하라.
"""

    response = llm.invoke(prompt)

    return response.content


# 대화 기록 저장소 만들기
if "messages" not in st.session_state:
    st.session_state.messages = []

if "stage" not in st.session_state:
    st.session_state.stage = "question"

if "original_question" not in st.session_state:
    st.session_state.original_question = ""

# 저장된 대화 다시 출력
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

#첫 대화 저장
if "first_answer" not in st.session_state:
    st.session_state.first_answer = ""

#첫 점수 대비 개선 점수 비교
if "first_score" not in st.session_state:
    st.session_state.first_score = 0

prompt = st.chat_input("궁금한 내용을 입력해 보세요!")

#레퍼런스 답 한번만 생성
if "reference_answer" not in st.session_state:
    st.session_state.reference_answer = ""


#전이 질문 저장 공간 추가
if "transfer_question" not in st.session_state:
    st.session_state.transfer_question = ""

if prompt:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    if st.session_state.stage == "question":

        st.session_state.original_question = prompt

        st.session_state.reference_answer = generate_reference_answer(
            prompt
        )
        
        answer = (
        "좋아. 바로 설명하기 전에 네 생각을 먼저 말해봐.\n\n"
        f"**질문:** {prompt}\n\n"
        "정확하지 않아도 괜찮아."
    )

        st.session_state.stage = "answer"


    elif st.session_state.stage == "answer":
# 첫 답변 저장
        st.session_state.first_answer = prompt

# 첫 답변 평가
        evaluation = evaluate_answer(
            st.session_state.original_question,
            st.session_state.reference_answer,
            prompt
        )
#첫 점수 계산
        final_score = calculate_score(evaluation)

        st.session_state.first_score = final_score

        level = get_level(final_score)
        
        if level == 5:

            next_question = generate_next_question(
                st.session_state.original_question,
                prompt,
                evaluation,
                level
            )

            answer = f"""
        ### 이해 수준
        Level 5

        이미 핵심 개념을 충분히 이해하고 있어요.

        ### 💭 다음 생각
        {next_question}
        """

            st.session_state.stage = "transfer"

        else:

            feedback = generate_feedback(
                st.session_state.original_question,
                prompt,
                evaluation,
                level
            )

            answer = f"""
        ### 현재 이해 수준
        Level {level}

        ### 피드백
        {feedback}

        ---

        ✏️ **피드백을 참고해서 처음 답변을 다시 작성해보세요.**
        """

            st.session_state.stage = "retry"
        

    elif st.session_state.stage == "retry":

        # 1. 수정 답변 다시 평가
        retry_evaluation = evaluate_answer(
            st.session_state.original_question,
            st.session_state.reference_answer,
            prompt
        )

        # 2. 수정 답변 점수 계산
        retry_score = calculate_score(
            retry_evaluation
        )

        # 3. 수정 답변 레벨 계산
        retry_level = get_level(
            retry_score
        )

        # 4. 처음 점수와 비교
        score_change = (
            retry_score
            - st.session_state.first_score
        )

        # 5. 다음 생각 질문 생성
        next_question = generate_next_question(
            st.session_state.original_question,
            prompt,
            retry_evaluation,
            retry_level
        )

        st.session_state.transfer_question = next_question

        # 6. 화면에 보여줄 답변
        answer = f"""
    ### 재평가 결과

    처음 점수: **{st.session_state.first_score}점**

    수정 후 점수: **{retry_score}점**

    변화: **{score_change:+.1f}점**

    현재 Level: **{retry_level}**

    ### 피드백
    {retry_evaluation["feedback"]}

    ---

    ### 💭 다음 생각

    {next_question}
    """

        # 7. 일단 다음 단계로 이동
        st.session_state.stage = "transfer"

    elif st.session_state.stage == "transfer":

        transfer_comment = generate_transfer_comment(
            st.session_state.original_question,
            st.session_state.transfer_question,
            prompt
        )

        answer = f"""
    ### 🧠 적용 결과

    {transfer_comment}

    ---

    새로운 질문이 있다면 자유롭게 입력해보세요.
    """

        st.session_state.stage = "question"
        
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    st.rerun()