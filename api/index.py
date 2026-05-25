from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Any, List, Dict
import os
import yaml
import traceback
import google.generativeai as genai
from datetime import datetime, timezone, timedelta

app = FastAPI()

# 💡 YAML 설정 동적 로드 함수 (Vercel 안전 경로)
def load_agent_config():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(base_dir, "agent_config.yaml")
    
    if not os.path.exists(yaml_path):
        # Fallback for some serverless environments
        yaml_path = os.path.join(os.getcwd(), "api", "agent_config.yaml")
        
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
    return {"message": "AHA API is running. Please use POST /api/chat to send messages."}

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # 1. YAML 환경 파일 로드
        config = load_agent_config()
        agents = config["agents"]
        
        user_input = request.message

        gemini_api_key = get_gemini_api_key()
        genai.configure(api_key=gemini_api_key)

        kst = timezone(timedelta(hours=9))
        current_time = datetime.now(kst)
        current_timestamp = current_time.strftime("%Y-%m-%dT%H:%M:%S KST")
        friendly_time_str = current_time.strftime("%p %I시 %M분").replace("AM", "오전").replace("PM", "오후")
        
        enriched_input = f"[시스템 현재 시각: {current_timestamp} ({friendly_time_str})] \n{user_input}"

        # --- Stage 1: 임상 데이터 추론 엔진 가동 ---
        analyzer_prompt = agents["clinical_reasoner"]["system_prompt"]
        model1 = genai.GenerativeModel("gemini-2.5-flash", system_instruction=analyzer_prompt)
        response_stage1 = model1.generate_content(
            f"복합 데이터 입력 (JSON 컨텍스트): {enriched_input}",
            generation_config={"temperature": 0.2}
        )
        analysis_result = response_stage1.text

        # --- Stage 2: 행동 변화 코치 에이전트 가동 (Context Chaining) ---
        coach_prompt = agents["behavior_coach"]["system_prompt"]
        model2 = genai.GenerativeModel("gemini-2.5-flash", system_instruction=coach_prompt)
        response_stage2 = model2.generate_content(
            f"임상 엔진 분석 로그:\n{analysis_result}\n\n위 추론 결과를 바탕으로, AHA 철학에 맞는 구체적이고 다정한 코칭 메시지를 작성해줘.",
            generation_config={"temperature": 0.6}
        )
        final_coach_answer = response_stage2.text

        # UI에 분석 로그와 최종 답변을 분리해서 전달
        # JSON 포맷을 예쁘게 보여주기 위해 코드블록 기호를 제거합니다
        clean_log = analysis_result.replace("```json", "").replace("```", "").strip()
        
        return {
            "analysis_log": clean_log.replace("\n", "<br>").replace(" ", "&nbsp;"),
            "answer": final_coach_answer
        }

    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_msg)