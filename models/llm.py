import logging
from groq import Groq
import google.genai as genai
from config.config import (
    GROQ_API_KEY, GEMINI_API_KEY,
    GROQ_MODEL, GEMINI_MODEL,
    SYSTEM_PROMPT, CONCISE_INSTRUCTION, DETAILED_INSTRUCTION
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


#  Groq LLM  (1st model)
class GroqLLM:

    def __init__(self):
        try:
            self.client = Groq(api_key=GROQ_API_KEY)
            self.model  = GROQ_MODEL
            logger.info("GroqLLM initialised with model: %s", self.model)
        except Exception as e:
            logger.error("Failed to initialise GroqLLM: %s", e)
            raise

    def generate(
        self,
        user_message: str,
        context: str = "",
        mode: str = "concise",
        chat_history: list = None,
    ) -> str:
        try:
            mode_instruction = (
                CONCISE_INSTRUCTION if mode == "concise" else DETAILED_INSTRUCTION
            )

            # Build system message
            system_content = f"{SYSTEM_PROMPT}\n\nResponse Style: {mode_instruction}"
            if context:
                system_content += f"\n\nRelevant Medical Context (from NIH/MedQuAD):\n{context}"

            messages = [{"role": "system", "content": system_content}]

            # Append conversation history
            if chat_history:
                messages.extend(chat_history)

            messages.append({"role": "user", "content": user_message})

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=1024 if mode == "concise" else 2048,
            )
            return response.choices[0].message.content

        except Exception as e:
            logger.error("GroqLLM generation error: %s", e)
            return f"⚠️ Groq Error: {str(e)}"


#  Gemini LLM  (Fallback / Large Context)
class GeminiLLM:

    def __init__(self):
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            self.model = genai.GenerativeModel(
                model_name=GEMINI_MODEL,
                system_instruction=SYSTEM_PROMPT,
            )
            logger.info("GeminiLLM initialised with model: %s", GEMINI_MODEL)
        except Exception as e:
            logger.error("Failed to initialise GeminiLLM: %s", e)
            raise

    def generate(
        self,
        user_message: str,
        context: str = "",
        mode: str = "concise",
        chat_history: list = None,
    ) -> str:
        try:
            mode_instruction = (
                CONCISE_INSTRUCTION if mode == "concise" else DETAILED_INSTRUCTION
            )

            # Build prompt with context and style
            prompt_parts = [f"Response Style: {mode_instruction}\n"]
            if context:
                prompt_parts.append(
                    f"Relevant Medical Context (from NIH/MedQuAD):\n{context}\n"
                )
            prompt_parts.append(f"User Question: {user_message}")
            full_prompt = "\n".join(prompt_parts)

            # Build Gemini history format
            history = []
            if chat_history:
                for msg in chat_history:
                    role = "user" if msg["role"] == "user" else "model"
                    history.append({"role": role, "parts": [msg["content"]]})

            chat_session = self.model.start_chat(history=history)
            response = chat_session.send_message(full_prompt)
            return response.text

        except Exception as e:
            logger.error("GeminiLLM generation error: %s", e)
            return f"⚠️ Gemini Error: {str(e)}"


# ──────────────────────────────────────────────────────────
#  Factory — get the right LLM
# ──────────────────────────────────────────────────────────

def get_llm(provider: str):
    try:
        if provider == "Groq (LLaMA 3.3)":
            return GroqLLM()
        elif provider == "Gemini (Flash)":
            return GeminiLLM()
        else:
            logger.warning("Unknown provider '%s', defaulting to Groq.", provider)
            return GroqLLM()
    except Exception as e:
        logger.error("get_llm failed for provider '%s': %s", provider, e)
        raise
