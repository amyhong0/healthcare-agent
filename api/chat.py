from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, List, Dict
import os
import yaml
import google.generativeai as genai

app = FastAPI()

# 💡 YAML 설정 동적 로드 함수
def load_agent_config():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(current_dir, "agent_config.yaml")
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = []


def get_gemini_api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not configured")
    return api_key


def extract_response_text(response: Any) -> str:
    if response is None:
        return ""
    if hasattr(response, "last"):
        last = response.last
        if hasattr(last, "message"):
            message = last.message
            if hasattr(message, "content"):
                content = message.content
                if isinstance(content, list) and content:
                    first = content[0]
                    if isinstance(first, dict):
                        return first.get("text", str(first))
                    return str(first)
                return str(content)
            return str(message)
    if hasattr(response, "candidates") and response.candidates:
        candidate = response.candidates[0]
        content = getattr(candidate, "content", None)
        if isinstance(content, list) and content:
            first = content[0]
            if isinstance(first, dict):
                return first.get("text", str(first))
            return str(first)
        return str(content)
    return str(response)


@app.post("/")
async def chat_endpoint(request: ChatRequest):
    try:
        # 1. YAML 환경 파일 로드
        config = load_agent_config()
        agents = config["agents"]
        
        user_input = request.message

        gemini_api_key = get_gemini_api_key()
        genai.configure(api_key=gemini_api_key)

        # --- Stage 1: 영양 분석 에이전트 가동 ---
        analyzer_prompt = agents["nutrition_analyzer"]["system_prompt"]
        response_stage1 = genai.chat.completions.create(
            model="gemini-1.5",
            messages=[
                {"author": "system", "content": analyzer_prompt},
                {"author": "user", "content": f"사용자 식단 기록: {user_input}"}
            ],
            temperature=0.3  # 분석은 정밀하게
        )
        analysis_result = extract_response_text(response_stage1)

        # --- Stage 2: 행동 변화 코치 에이전트 가동 (Context Chaining) ---
        coach_prompt = agents["behavior_coach"]["system_prompt"]
        response_stage2 = genai.chat.completions.create(
            model="gemini-1.5",
            messages=[
                {"author": "system", "content": coach_prompt},
                {
                    "author": "user",
                    "content": f"영양 분석 에이전트의 결과:\n{analysis_result}\n\n이 결과를 바탕으로 사용자에게 친절한 개인화 웰다 코칭 문장을 작성해줘."
                }
            ],
            temperature=0.7  # 코칭은 풍부하고 다정하게
        )
        final_coach_answer = extract_response_text(response_stage2)

        # 웰다 UI에 분석 로그와 최종 답변을 분리해서 전달
        return {
            "analysis_log": analysis_result.replace("\n", "<br>"),
            "answer": final_coach_answer
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
