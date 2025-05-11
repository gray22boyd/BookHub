import os
from openai import OpenAI
from dotenv import load_dotenv
from agents.answer_agent import AnswerAgent
from agents.organizer_agent import OrganizerAgent

# Load environment variables and set OpenAI API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    print("⚠️ OpenAI key not loaded from .env file")

# Initialize the OpenAI client
client = OpenAI(api_key=openai_api_key)

class LeadAgent:
    def __init__(self):
        self.answer_agent = AnswerAgent()
        self.organizer_agent = OrganizerAgent()
        # Verify API key on initialization
        if not openai_api_key:
            print("⚠️ WARNING: OpenAI API key not found in environment variables")

    def classify_prompt(self, prompt: str) -> str:
        if any(keyword in prompt.lower() for keyword in ["summarize", "analyze", "explain"]):
            return "answer"
        elif any(keyword in prompt.lower() for keyword in ["add", "upload", "ingest"]):
            return "organize"
        else:
            return "general"

    def handle_prompt(self, prompt: str) -> str:
        classification = self.classify_prompt(prompt)
        if classification == "answer":
            return self.answer_agent.handle(prompt)
        elif classification == "organize":
            return self.organizer_agent.handle(prompt)
        elif classification == "general":
            if not openai_api_key:
                return "⚠️ OpenAI API key is missing or invalid. Please check your .env file and restart the application."
            return self.general_chat_response(prompt)
        else:
            return self.answer_agent.handle(prompt)  # Default to AnswerAgent for unknown classifications
    
    def general_chat_response(self, prompt: str) -> str:
        """
        Send prompt to OpenAI and return the response
        """
        if not openai_api_key:
            return "⚠️ OpenAI API key is missing or invalid. Please check your .env file and restart the application."
            
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are BookHub, an AI book companion that helps users with questions about books, authors, and literature in general."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            error_message = str(e)
            print(f"⚠️ Error calling OpenAI API: {error_message}")
            
            if "authentication" in error_message.lower() or "api key" in error_message.lower():
                return "Sorry, there seems to be an issue with my API key. Please check your configuration and try again."
            elif "rate limit" in error_message.lower():
                return "I'm experiencing high demand right now. Please try again in a moment."
            else:
                return "Sorry, I'm having trouble connecting to OpenAI. Please try again later or check your API key." 