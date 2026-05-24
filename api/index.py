import os
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional
import time

app = FastAPI()

# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Route to serve frontend HTML directly from root
@app.get("/")
@app.get("/index.html")
def read_root():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    paths_to_try = [
        os.path.join(current_dir, "..", "public", "index.html"),
        os.path.join(current_dir, "public", "index.html"),
        os.path.join(os.getcwd(), "public", "index.html"),
    ]
    for path in paths_to_try:
        if os.path.exists(path):
            return FileResponse(path)
    return HTMLResponse("<h1>대웅 웰다 프론트엔드를 찾을 수 없습니다.</h1>", status_code=404)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    agent_id: str
    messages: List[Message]

# Load agents configuration
def load_agents():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Try parent directory then current directory
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
    
    # Fallback default agents if config not found
    return [
        {
            "id": "welda_coach",
            "name": "웰다 케어 코치",
            "role": "Wellness & Health Coach",
            "description": "대웅 웰다의 메인 건강 코칭 에이전트입니다. 식단, 운동, 영양제 등 개인화된 웰니스 가이드를 제공합니다.",
            "welcome_message": "안녕하세요! 당신의 개인 맞춤 웰니스 코치, 웰다입니다. 오늘 하루는 어떻게 보내셨나요? 무엇이든 편하게 물어보세요!",
            "system_prompt": "당신은 대웅제약의 웰니스 케어 서비스 '웰다(Welda)'의 전문 AI 웰니스 코치입니다.",
            "color": "#4F46E5",
            "gradient": "linear-gradient(135deg, #818CF8, #4F46E5)"
        }
    ]

# Get agents list
@app.get("/api/agents")
def get_agents():
    agents = load_agents()
    # Return agents metadata (omit system prompt for security/cleanliness if needed, but keeping it is fine)
    return agents

# Chat endpoint
@app.post("/api/chat")
async def chat(request: ChatRequest):
    agents = load_agents()
    selected_agent = next((a for a in agents if a["id"] == request.agent_id), None)
    
    if not selected_agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    user_message = request.messages[-1].content if request.messages else ""
    
    # Check for GEMINI_API_KEY environment variable to call Gemini API
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            
            # System prompt configuration
            system_instruction = selected_agent.get("system_prompt", "")
            model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                system_instruction=system_instruction
            )
            
            # Convert message history for Gemini API
            contents = []
            for msg in request.messages[:-1]:
                role = "user" if msg.role == "user" else "model"
                contents.append({"role": role, "parts": [msg.content]})
            
            # Append current user message
            contents.append({"role": "user", "parts": [user_message]})
            
            response = model.generate_content(contents)
            return {"content": response.text}
        except Exception as e:
            print(f"Gemini API Error: {e}. Falling back to simulation mode.")
            # Fall through to simulation if API fails
            pass

    # Simulated Smart Response Mode (Local fallback)
    response_text = generate_simulated_response(selected_agent["id"], user_message, request.messages)
    
    # Add a slight delay to simulate thinking time for realism
    time.sleep(0.6)
    
    return {"content": response_text}

def generate_simulated_response(agent_id: str, user_message: str, history: List[Message]) -> str:
    user_msg_clean = user_message.strip().lower()
    
    if agent_id == "welda_coach":
        if "식단" in user_msg_clean or "밥" in user_msg_clean or "먹었" in user_msg_clean:
            return "오늘 드신 식단이 궁금하네요! 균형 잡힌 영양 섭취를 위해 탄수화물, 단백질, 지방의 비율이 조화를 이루었는지 점검해드릴게요. 구체적으로 어떤 음식을 드셨는지 알려주시면, 저희 '영양분석 마스터' 에이전트와 연계하여 더 정밀하게 분석해 드릴 수도 있습니다."
        elif "운동" in user_msg_clean or "헬스" in user_msg_clean or "달리기" in user_msg_clean or "피트니스" in user_msg_clean:
            return "멋진 시도입니다! 규칙적인 신체 활동은 기초대사량을 높이고 혈당을 안정화하는 데 최고의 방법입니다. 오늘은 유산소 운동 위주로 하셨나요, 아니면 근력 운동 위주로 하셨나요? 본인의 체력 수준에 맞춰 무리하지 않는 선에서 지속 가능한 루틴을 만들어 드릴게요."
        elif "피곤" in user_msg_clean or "졸려" in user_msg_clean or "피로" in user_msg_clean:
            return "만성 피로는 수면 질, 불균형한 식습관, 또는 특정 영양소 결핍 때문에 나타날 수 있어요. 최근 수면 시간은 충분하셨나요? 또한 비타민 B군이나 마그네슘 같은 에너지 대사 관련 영양소 섭취 상태도 점검해 보시면 좋겠습니다. 영양제 섭취 상태가 궁금하시다면 언제든 말씀해 주세요."
        elif "영양제" in user_msg_clean or "비타민" in user_msg_clean:
            return "건강 유지를 위해 영양제를 챙겨 드시는 것은 훌륭한 습관입니다! 복용 중이신 영양제의 종류와 복용 시간을 말씀해 주시면, 흡수율을 높이는 섭취 타이밍이나 혹시 겹쳐서 과다 복용 중인 성분이 있는지 '영양제 큐레이터' 에이전트의 데이터베이스를 바탕으로 꼼꼼히 확인해 드릴게요."
        else:
            return f"보내주신 의견 감사합니다. 대웅 웰다는 단순한 건강 관리를 넘어, 과학적이고 지속 가능한 라이프스타일을 제안합니다. 혹시 최근 가장 집중해서 관리하고 싶으신 건강 목표(체중 감량, 혈당 케어, 피로 회복 등)가 무엇인지 말씀해 주시겠어요?"
            
    elif agent_id == "diet_expert":
        if "삼겹살" in user_msg_clean or "고기" in user_msg_clean:
            return "삼겹살은 훌륭한 단백질과 지방 공급원이지만, 포화지방 비율이 높으므로 쌈채소(상추, 깻잎)와 마늘, 양파를 듬뿍 곁들여 식이섬유와 항산화 물질을 함께 섭취하는 것이 혈당과 혈관 건강에 좋습니다. 밥은 백미밥 대신 현미밥이나 곤약밥으로 반 공기만 곁들이는 저탄고지 방식을 추천해 드립니다."
        elif "샐러드" in user_msg_clean:
            return "닭가슴살이나 삶은 계란 등 단백질 토핑이 들어간 샐러드는 혈당 조절과 다이어트에 최적의 식단입니다! 다만, 드레싱에 당류가 많이 포함되어 있을 수 있으니 발사믹이나 올리브유 기반 드레싱을 선택하시는 것을 추천해 드립니다. 오늘 식사는 완벽한 혈당 스파이크 방지 식단이네요!"
        elif "라면" in user_msg_clean or "짜장면" in user_msg_clean or "밀가루" in user_msg_clean or "면" in user_msg_clean:
            return "정제 탄수화물인 밀가루 면 요리는 혈당을 급격하게 올리는 '혈당 스파이크'의 주범입니다. 어쩔 수 없이 드셔야 한다면, 면을 드시기 전에 계란이나 오이, 두부 같은 단백질/식이섬유를 먼저 드셔서 소화 흡수 속도를 늦춰주세요. 다음 식사 때는 식이섬유와 단백질 위주로 보완하는 식단을 권장합니다."
        else:
            return "오늘 식사하신 메뉴(식재료, 양 등)를 자세히 알려주시면, 대략적인 칼로리와 탄수화물/단백질/지방(탄단지) 배합 비율을 분석해 드릴게요. 혈당 스파이크 걱정 없는 건강한 대안 식단도 함께 제안해 드리겠습니다!"

    elif agent_id == "supplement_expert":
        if "오메가" in user_msg_clean or "오메가3" in user_msg_clean:
            return "오메가3는 지용성 성분이기 때문에 공복에 드시면 흡수율이 크게 떨어집니다. 따라서 기름진 음식을 드시는 점심이나 저녁 식사 직후에 바로 복용하시는 것이 가장 좋습니다. 또한 항응고제를 복용 중이시라면 혈액 순환을 과도하게 촉진할 수 있으니 주의가 필요합니다."
        elif "유산균" in user_msg_clean or "프로바이오틱스" in user_msg_clean:
            return "유산균(프로바이오틱스)은 위산에 약하기 때문에 위산 분비가 비교적 적은 아침 공복에 충분한 물과 함께 복용하시는 것이 장까지 살아가는 데 유리합니다. 간혹 위장 장애가 생기신다면 가벼운 식사 후 드시는 것으로 바꾸셔도 좋습니다."
        elif "비타민c" in user_msg_clean or "종합비타민" in user_msg_clean:
            return "비타민 C는 강력한 항산화제로 면역력과 피로 회복에 도움을 주지만, 산성이 강해 공복에 복용 시 위 쓰림을 유발할 수 있습니다. 위가 예민하신 편이라면 반드시 아침 또는 점심 식사 중에나 식후 직후에 복용해 주세요. 또한 고용량 복용 시 충분한 수분 섭취가 필수적입니다."
        else:
            return "현재 복용 중이시거나 복용을 고민 중이신 영양제들을 모두 말씀해 주시면, 최적의 복용 시간대 배치와 상호작용(궁합), 그리고 과다 섭취 중인 성분이 있는지 정밀 분석해 드리겠습니다."
            
    return "안녕하세요! 대웅 웰다 AI 코칭 서비스입니다. 질문해주신 내용에 대해 심도 있게 검토하여 웰니스 케어에 도움이 되는 답변을 드리겠습니다. 더 궁금한 점이 있으시면 편하게 적어주세요."
