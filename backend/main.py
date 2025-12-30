from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai

# ---------------------------
# CONFIGURE GEMINI API
# ---------------------------
API_KEY = 'AIzaSyAKW0e42DZNfmcpqR6IhIadBGlqbdzmLgA'
genai.configure(api_key=API_KEY)

# ---------------------------
# CREATE FASTAPI APP
# ---------------------------
app = FastAPI(title='Buddy Chatbot')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],  # Allow all origins
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# ---------------------------
# REQUEST MODEL
# ---------------------------
class ChatRequest(BaseModel):
    message: str

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------
def analyze_question_type(question: str):
    q = question.lower()
    if len(q.split()) <= 4:
        return "beginner"
    if "why" in q or "how" in q:
        return "explanatory"
    if any(w in q for w in ["api", "model", "algorithm", "neural", "framework"]):
        return "technical"
    return "general"

def detect_intent(question: str):
    q = question.lower()
    if "what is" in q or "define" in q:
        return "definition"
    if "how" in q or "steps" in q:
        return "problem_solving"
    if "why" in q:
        return "learning"
    return "exploration"

def confidence_score(question: str):
    score = 0.3
    if len(question.split()) > 6:
        score += 0.3
    if any(w in question.lower() for w in ["api", "model", "algorithm"]):
        score += 0.2
    if "?" in question:
        score += 0.2
    return min(score, 1.0)

def generate_prompt(question, q_type, intent, confidence):
    style = "simple explanation"
    if confidence > 0.7:
        style = "technical explanation"
    elif confidence > 0.5:
        style = "step-by-step explanation"

    return f"""
You are Buddy, an intelligent AI assistant.

User Intent: {intent}
Question Type: {q_type}
Confidence Level: {confidence}

Instruction:
Respond with a {style} adapted to the user's intent.

Question:
{question}
"""

def buddy_fallback(question):
    """Dynamic fallback response if Gemini API fails"""
    q = question.lower()
    if "hi" in q or "hello" in q:
        return "Hello! I am Buddy 🤖. How can I help you today?"
    if "api" in q:
        return "API stands for Application Programming Interface."
    if "model" in q:
        return "A model is a representation used by AI to make predictions or generate content."
    if "python" in q:
        return "Python is a popular programming language used in AI, web development, and more."
    return f"You asked: {question}. I'm learning new things every day!"

# ---------------------------
# CHAT ENDPOINT
# ---------------------------
@app.post("/chat")
def chat(request: ChatRequest):
    user_message = request.message

    q_type = analyze_question_type(user_message)
    intent = detect_intent(user_message)
    confidence = confidence_score(user_message)
    prompt = generate_prompt(user_message, q_type, intent, confidence)

    try:
        # Try Gemini API
        model = genai.GenerativeModel("gemini-1.5-mini")
        response = model.generate_content(prompt)
        answer = response.text

    except Exception as e:
        # Fallback to dynamic response
        answer = buddy_fallback(user_message)

    return {
        "response": answer,
        "question_type": q_type,
        "intent": intent,
        "confidence": confidence
    }
