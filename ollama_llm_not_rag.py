from ollama import chat
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain_community.llms import Ollama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

import argparse
import time

parser = argparse.ArgumentParser()
# 추가 인자 필요 시 여기에 추가
args = parser.parse_args()

def process_time_cal(task_name, start_time):
    end_time = time.process_time()
    print(f"task_name: {task_name}")
    print(f"time elapsed : {int(round((end_time - start_time) * 1000))}ms")

if __name__ == "__main__":
    
    # 스트리밍 콜백 핸들러 생성: 토큰이 생성되는 즉시 stdout에 출력함
    streaming_handler = StreamingStdOutCallbackHandler()
    
    # LLM 초기화 (스트리밍 모드 활성화)
    start_time = time.process_time()
    llm = Ollama(model="deepseek_r1_14b", callbacks=[streaming_handler])
    process_time_cal("Ollama(model=deepseek_r1_14b)", start_time)
    
    # 프롬프트 템플릿 구성: 미리 정의한 시나리오(문맥)와 질문을 입력받음.
    prompt = """
                Use the following context to answer the question:
                Context: {context}
                Question: {question}
                Answer(번호만 출력):
            """
    start_time = time.process_time()
    QA_PROMPT = PromptTemplate.from_template(prompt)
    print(f"\nQA_PROMPT: {QA_PROMPT}\n")
    process_time_cal("PromptTemplate.from_template", start_time)
    
    # LLM 체인 생성
    start_time = time.process_time()
    llm_chain = LLMChain(llm=llm, prompt=QA_PROMPT)
    process_time_cal("LLMChain(llm=llm, prompt=QA_PROMPT)", start_time)
    
    # 미리 정의한 시나리오 텍스트 (번호와 문장이 같이 있음)
    scenario_context = (
        "1. 환자분 안녕하세요\n"
        "2. 환자분 성함이 어떻게 되세요\n"
        "3. 어디가 불편하세요\n"
        "4. 환자분 진료 시작하겠습니다\n"
        "5. 환자분 아 하세요\n"
        "6. 환자분 입을 더 크게 벌려보시겠어요\n"
        "7. 통증이 느껴진다면 왼쪽 팔을 들어주세요\n"
        "8. 환자분 통증이 느껴지시나요\n"
        "9. 환자분 다시 진료 시작해도 될까요\n"
        "10. 환자분 진료 끝났습니다. 수고하셨습니다"
    )
    
    # 사용자로부터 지속적으로 질문을 입력받아 스트리밍 방식으로 답변 출력
    print("\n스트리밍 모드: 질문을 입력하세요. 종료하려면 'exit' 또는 'quit'을 입력하세요.\n")
    while True:
        question = input("Question: ")
        start_time = time.process_time()
        if question.lower() in ["exit", "quit"]:
            break

        print("Response:")
        # llm_chain.run 호출 시, 프롬프트 내 {context}와 {question} 자리에 각각 시나리오와 사용자의 입력이 채워짐.
        response = llm_chain.run({"context": scenario_context, "question": question})
        print(f"response!!!!!!!!!!!!!! \n{response}\n")
        # 스트리밍 콜백에 의해 토큰들이 실시간 출력되므로, 최종 응답도 함께 표시됨.
        print("\n")
        process_time_cal("Event Function processing times", start_time)

