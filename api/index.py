from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import os
import yaml
from openai import OpenAI

app = FastAPI()

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 💡 YAML 설정 동적 로드 함수
def load_agent_config():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(current_dir, "agent_config.yaml")
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = []

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # 1. YAML 환경 파일 로드
        config = load_agent_config()
        agents = config["agents"]
        
        user_input = request.message

        # --- Stage 1: 영양 분석 에이전트 가동 ---
        analyzer_prompt = agents["nutrition_analyzer"]["system_prompt"]
        response_stage1 = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": analyzer_prompt},
                {"role": "user", "content": f"사용자 식단 기록: {user_input}"}
            ],
            temperature=0.3 # 분석은 정밀하게
        )
        analysis_result = response_stage1.choices[0].message.content

        # --- Stage 2: 행동 변화 코치 에이전트 가동 (Context Chaining) ---
        coach_prompt = agents["behavior_coach"]["system_prompt"]
        response_stage2 = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": coach_prompt},
                {"role": "user", "content": f"영양 분석 에이전트의 결과:\n{analysis_result}\n\n이 결과를 바탕으로 사용자에게 친절한 개인화 웰다 코칭 문장을 작성해줘."}
            ],
            temperature=0.7 # 코칭은 풍부하고 다정하게
        )
        final_coach_answer = response_stage2.choices[0].message.content

        # 웰다 UI에 분석 로그와 최종 답변을 분리해서 전달
        return {
            "analysis_log": analysis_result.replace("\n", "<br>"),
            "answer": final_coach_answer
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))