from typing import Dict, List
from openai import OpenAI

def generate_response(openai_key: str, user_message: str, context: str, 
                     conversation_history: List[Dict], model: str = "gpt-3.5-turbo") -> str:
    """Generate response using OpenAI with context"""
    system_prompt = f"""You are a NASA document assistant.
        - Carefully read the context below and answer based on the context.
        - If you cannot find proper information in the documents provided the response with "Apologies, but I don't have enough information to answer that"
        - Do not make up or hallucinate any information or search through outside sources to produce information.
        - Be clear and academic in your responses, keep things brief and concise.
    """

    # Used similar code that reviewer was using for help 02/11/2026
    messages = [{"role": "system", "content": system_prompt}] + conversation_history

    user_message_with_context = f"""NASA documents context:
        Context: {context}
        Question: {user_message}
    """

    messages.append({"role": "user", "content": user_message_with_context})

    client = OpenAI(
        api_key=openai_key,
        base_url="https://openai.vocareum.com/v1"
    )
    
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )

    conversation_history.extend({"role": "assistant", "content": response})
    print(response)
    return response.choices[0].message.content