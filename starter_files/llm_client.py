from typing import Dict, List
from openai import OpenAI

def generate_response(openai_key: str, user_message: str, context: str, 
                     conversation_history: List[Dict], model: str = "gpt-3.5-turbo") -> str:
    """Generate response using OpenAI with context"""
    system_prompt = f"""You are a NASA document assistant.
        Carefully read the context below and answer based on the chromadb exposure.

        
        Context: {context}

        Conversation: {conversation_history}
    """
    conversation_history.append(user_message)
    client = OpenAI(
        api_key=openai_key,
        base_url="https://openai.vocareum.com/v1"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    )

    return response.choices[0].message.content