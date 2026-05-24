import os
import yaml
import datetime
import time
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS for frontend requests
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


def load_agents() -> List[Dict[str, Any]]:
	current_dir = os.path.dirname(os.path.abspath(__file__))
	paths_to_try = [
		os.path.join(current_dir, "..", "config", "agents.yaml"),
		os.path.join(current_dir, "config", "agents.yaml"),
		os.path.join(os.getcwd(), "config", "agents.yaml"),
	]
	for path in paths_to_try:
		if os.path.exists(path):
			try:
				with open(path, "r", encoding="utf-8") as f:
					data = yaml.safe_load(f)
					return data.get("agents", [])
			except Exception as e:
				print(f"Error loading yaml from {path}: {e}")

	# Fallback default agents
	return [
		{
			"id": "welda_coach",
			"name": "웰다 케어 코치",
			"role": "Wellness & Health Coach",
			"description": "대웅 웰다의 메인 건강 코칭 에이전트입니다.",
			"welcome_message": "안녕하세요! 웰다입니다.",
			"system_prompt": "당신은 웰다 케어 코치입니다."
		},
		{
			"id": "diet_expert",
			"name": "영양분석 마스터",
			"role": "Dietary & Nutrition Specialist",
			"description": "식단 기반 분석 에이전트",
			"system_prompt": "당신은 영양 전문가입니다."
		},
		{
			"id": "supplement_expert",
			"name": "영양제 큐레이터",
			"role": "Supplement Specialist",
			"description": "영양제 추천 에이전트",
			"system_prompt": "당신은 영양 보충제 전문가입니다."
		}
	]


@app.get("/api/agents")
def get_agents():
	return load_agents()


@app.post("/api/chat")
async def chat(payload: Dict[str, Any]):
	"""Expecting payload: { agent_id: str, messages: [{role, content}, ...] }

	Returns JSON: { analysis_log: str (HTML), answer: str, content: str }
	"""
	agent_id = payload.get("agent_id")
	messages = payload.get("messages", [])

	if not agent_id:
		raise HTTPException(status_code=400, detail="agent_id is required")

	agents = load_agents()
	selected = next((a for a in agents if a.get("id") == agent_id), None)
	if not selected:
		raise HTTPException(status_code=404, detail="Agent not found")

	user_message = ""
	if messages and isinstance(messages, list):
		last = messages[-1]
		user_message = last.get("content", "")

	# Try calling external model if key present
	gem_key = os.environ.get("GEMINI_API_KEY")
	if gem_key:
		try:
			# import inside to avoid import-time errors on environments without package
			import google.generativeai as genai
			genai.configure(api_key=gem_key)
			system_prompt = selected.get("system_prompt", "")
			model = genai.GenerativeModel(model_name="gemini-1.5-flash", system_instruction=system_prompt)

			contents = []
			for msg in messages[:-1]:
				role = "user" if msg.get("role") == "user" else "model"
				contents.append({"role": role, "parts": [msg.get("content", "")]})
			contents.append({"role": "user", "parts": [user_message]})

			resp = model.generate_content(contents)
			answer_text = getattr(resp, "text", str(resp))
			analysis_log = build_analysis_log(selected, user_message, "API model used: gemini-1.5-flash")
			return {"analysis_log": analysis_log, "answer": answer_text, "content": answer_text}
		except Exception as e:
			print(f"Gemini call failed: {e}")

	# Fallback simulated response
	answer_text = generate_simulated_response(agent_id, user_message)
	analysis_log = build_analysis_log(selected, user_message, "Simulated analysis: tokenization → parsing → scoring → recommendation")
	# small delay to mimic processing
	time.sleep(0.4)
	return {"analysis_log": analysis_log, "answer": answer_text, "content": answer_text}


def build_analysis_log(agent: Dict[str, Any], user_message: str, summary: str) -> str:
	ts = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
	agent_name = agent.get("name", agent.get("id", "agent"))
	role = agent.get("role", "")
	steps = [
		"수신: 입력 수신 및 전처리",
		"엔티티 추출: 음식/영양소/양 파싱",
		"계산: 칼로리 및 탄단지(탄수/단백/지방) 추정",
		"평가: 혈당 스파이크 위험 점수 산출",
		"추천: 대체 식단 및 섭취 팁 생성"
	]
	items = '\n'.join([f"<li><strong>Step {i+1}:</strong> {s}</li>" for i, s in enumerate(steps)])
	html = f"<div class='agent-log-inner'><div><strong>{agent_name} ({role})</strong> · {ts}</div><div style='margin-top:6px;color:#374151;font-size:13px'>{summary}</div><ol style='margin-top:8px;padding-left:18px;color:#374151'>{items}</ol></div>"
	return html


def generate_simulated_response(agent_id: str, user_message: str) -> str:
	m = (user_message or "").lower()
	if agent_id == "welda_coach":
		if any(k in m for k in ["식단", "밥", "먹었"]):
			return "오늘 드신 식단을 알려주시면 영양분석 마스터와 연계하여 상세 분석을 제공하겠습니다."
		if any(k in m for k in ["운동", "헬스", "달리기"]):
			return "운동 루틴에 맞춘 권장 전략을 제안해드릴게요. 유산소/무산소 중 어떤 운동을 하셨나요?"
		return "도움을 드리겠습니다. 어떤 건강 목표를 가지고 계신가요?"

	if agent_id == "diet_expert":
		if any(k in m for k in ["삼겹살", "고기"]):
			return "삼겹살은 단백질과 지방이 풍부합니다. 상추와 채소를 곁들이면 혈당 관리에 도움이 됩니다."
		if "샐러드" in m:
			return "단백질 토핑이 포함된 샐러드는 균형 잡힌 선택입니다. 드레싱의 당 함량을 주의하세요."
		return "메뉴를 입력해 주시면 칼로리와 영양 구성을 분석해 드립니다."

	if agent_id == "supplement_expert":
		if any(k in m for k in ["오메가", "오메가3"]):
			return "오메가3는 식후 복용을 권장합니다. 항응고제 복용 시 주의하세요."
		if any(k in m for k in ["유산균", "프로바이오틱스"]):
			return "유산균은 공복에 물과 함께 복용하면 장까지 도달하기 좋습니다."
		return "복용 중인 영양제 목록을 알려주시면 상호작용을 점검해 드리겠습니다."

	return "안녕하세요. 웰다입니다. 무엇을 도와드릴까요?"


