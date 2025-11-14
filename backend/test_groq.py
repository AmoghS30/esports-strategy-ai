# backend/test_groq.py
from llm_engine import ESportsLLM

def test():
    print("Testing Groq API...\n")
    
    llm = ESportsLLM()
    
    prompt = "Briefly analyze: Team A secured early Baron at 20 minutes and pushed to win. What was the key turning point?"
    
    print("\nPrompt:", prompt)
    print("\n" + "="*50)
    
    response = llm.generate_response(prompt, max_length=200)
    
    print("Response:", response)
    print("="*50)

if __name__ == "__main__":
    test()