from groq import Groq
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("API_KEY")

# Initialize Groq Client
client = Groq(api_key=api_key)

# Store conversation history for better context
messages = [
    {
        "role": "system",
        "content": (
            "You are Fareed's AI, a highly knowledgeable and empathetic health assistant. "
            "Your main expertise is hypertrophy, fitness, and overall well-being. "
            "Answer user queries in a friendly, motivating, and informative manner. "
            "Provide science-backed advice, and if necessary, suggest actionable steps, routines, or diet plans. "
            "Avoid making medical diagnoses but encourage consulting healthcare professionals when appropriate."
        )
    }
]

while True:
    # Get user input
    user_input = input("\nUser: ")

    # Exit condition
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Fareed's AI: Take care! Stay healthy. 💪")
        break

    # Append user's message to maintain context
    messages.append({"role": "user", "content": user_input})

    # Generate AI response
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.7,
        max_completion_tokens=300,
        top_p=0.9,  # Set this higher to allow more variation in responses
        stream=True,
    )

    # Collect and print response
    assistant_response = ""
    print("Fareed's AI: ", end="")
    for chunk in completion:
        content = chunk.choices[0].delta.content or ""
        print(content, end="", flush=True)
        assistant_response += content

    print("\n")  # New line for better readability

    # Append assistant's message to maintain context
    messages.append({"role": "assistant", "content": assistant_response})
