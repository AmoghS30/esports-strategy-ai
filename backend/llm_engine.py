# backend/llm_engine.py
import requests
import os
from dotenv import load_dotenv
import time

load_dotenv()

class ESportsLLM:
    def __init__(self):
        """
        Use Groq API - Fast, reliable, and free!
        Updated with current model names (as of Nov 2024)
        """
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.api_key = os.getenv('GROQ_API_KEY', '')
        
        if not self.api_key:
            print("❌ GROQ_API_KEY not found in .env file")
            print("   Get your free API key at: https://console.groq.com/")
            raise ValueError("GROQ_API_KEY is required")
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Current available models (Nov 2024)
        self.models = {
            "llama-3.1-70b": "llama-3.1-70b-versatile",     # Best quality
            "llama-3.1-8b": "llama-3.1-8b-instant",         # Fast & good
            "llama-3.2-3b": "llama-3.2-3b-preview",         # Lightweight
            "mixtral": "mixtral-8x7b-32768",                # Very good
            "gemma2-9b": "gemma2-9b-it"                     # Alternative
        }
        
        # Use the fast model by default
        self.current_model = self.models["llama-3.1-8b"]
        
        print("✅ Using Groq API (Fast & Free!)")
        print(f"📦 Model: {self.current_model}")
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self):
        """Test if API key works"""
        try:
            test_payload = {
                "model": self.current_model,
                "messages": [
                    {"role": "user", "content": "Hello"}
                ],
                "max_tokens": 10
            }
            
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=test_payload,
                timeout=10
            )
            
            if response.status_code == 200:
                print("✅ API connection successful")
                print("✅ System ready!")
                return True
            else:
                print(f"⚠️  API test failed: {response.status_code}")
                print(f"Response: {response.text}")
                
                # Try to auto-fix by listing available models
                if response.status_code == 400:
                    print("\n🔄 Attempting to fetch available models...")
                    self._get_available_models()
                
                return False
                
        except Exception as e:
            print(f"⚠️  Connection test error: {e}")
            return False
    
    def _get_available_models(self):
        """Fetch list of available models from Groq"""
        try:
            models_url = "https://api.groq.com/openai/v1/models"
            response = requests.get(
                models_url,
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                models_data = response.json()
                available_models = [model["id"] for model in models_data.get("data", [])]
                
                if available_models:
                    print("\n📋 Available models:")
                    for model in available_models:
                        print(f"   - {model}")
                    
                    # Auto-select first available model
                    self.current_model = available_models[0]
                    print(f"\n✅ Auto-selected: {self.current_model}")
                    
        except Exception as e:
            print(f"⚠️  Could not fetch models: {e}")
    
    def generate_response(self, prompt, max_length=1024, temperature=0.7, retry_count=3):
        """
        Generate response using Groq API
        Super fast - usually responds in 1-3 seconds!
        """
        
        payload = {
            "model": self.current_model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a professional esports analyst with deep knowledge of competitive gaming. Provide detailed, insightful analysis and strategic recommendations."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": temperature,
            "max_tokens": max_length,
            "top_p": 0.9
        }
        
        for attempt in range(retry_count):
            try:
                print(f"🔄 Generating response (attempt {attempt + 1}/{retry_count})...")
                start_time = time.time()
                
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=60
                )
                
                elapsed_time = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    text = result["choices"][0]["message"]["content"]
                    
                    print(f"✅ Response generated in {elapsed_time:.2f}s")
                    return text
                
                elif response.status_code == 400:
                    error_data = response.json()
                    error_msg = error_data.get("error", {}).get("message", "Unknown error")
                    
                    if "decommissioned" in error_msg or "not supported" in error_msg:
                        print(f"⚠️  Model {self.current_model} is no longer available")
                        print("🔄 Fetching available models...")
                        self._get_available_models()
                        
                        if attempt < retry_count - 1:
                            print("Retrying with new model...")
                            continue
                    
                    return f"Error: {error_msg}"
                
                elif response.status_code == 429:
                    print("⚠️  Rate limited, waiting 5 seconds...")
                    time.sleep(5)
                    continue
                
                elif response.status_code == 401:
                    return "Error: Invalid Groq API key. Please check your .env file."
                
                else:
                    print(f"❌ API Error: {response.status_code}")
                    error_text = response.text[:300]
                    print(f"Response: {error_text}")
                    
                    if attempt < retry_count - 1:
                        print("Retrying in 3 seconds...")
                        time.sleep(3)
                        continue
                    
                    return f"Error: API returned status code {response.status_code}"
            
            except requests.exceptions.Timeout:
                print(f"⏱️  Request timed out")
                if attempt < retry_count - 1:
                    time.sleep(3)
                    continue
                return "Error: Request timed out. Please try again."
            
            except requests.exceptions.RequestException as e:
                print(f"❌ Network error: {e}")
                if attempt < retry_count - 1:
                    time.sleep(3)
                    continue
                return f"Error: Network issue - {str(e)}"
            
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                return f"Error: {str(e)}"
        
        return "Error: Failed to generate response after multiple attempts."
    
    def switch_model(self, model_name):
        """Switch to a different Groq model"""
        if model_name in self.models:
            self.current_model = self.models[model_name]
            print(f"✅ Switched to model: {self.current_model}")
        else:
            print(f"⚠️  Unknown model: {model_name}")
            print(f"Available: {list(self.models.keys())}")