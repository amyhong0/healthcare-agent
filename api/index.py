from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
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


@app.get("/")
async def serve_ui():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(current_dir, "..", "index.html")
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>UI file not found.</h1>", status_code=404)

@app.get("/api/chat")
async def root_get():
    return {"message": "Welda API is running. Please use POST /api/chat to send messages."}

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # 1. YAML 환경 파일 로드
        config = load_agent_config()
        agents = config["agents"]
        
        user_input = request.message

        gemini_api_key = get_gemini_api_key()
        genai.configure(api_key=gemini_api_key)

        # --- Stage 1: 임상 데이터 추론 엔진 가동 ---
        analyzer_prompt = agents["clinical_reasoner"]["system_prompt"]
        response_stage1 = genai.chat.completions.create(
            model="gemini-1.5",
            messages=[
                {"author": "system", "content": analyzer_prompt},
                {"author": "user", "content": f"사용자 복합 데이터 입력 (JSON 컨텍스트): {user_input}"}
            ],
            temperature=0.2  # 정확한 임상 추론
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
                    "content": f"임상 엔진 분석 로그:\n{analysis_result}\n\n위 추론 결과를 바탕으로, 웰다 철학에 맞는 구체적이고 다정한 코칭 메시지를 작성해줘."
                }
            ],
            temperature=0.6
        )
        final_coach_answer = extract_response_text(response_stage2)

        # 웰다 UI에 분석 로그와 최종 답변을 분리해서 전달
        return {
            "analysis_log": analysis_result.replace("\n", "<br>"),
            "answer": final_coach_answer
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))