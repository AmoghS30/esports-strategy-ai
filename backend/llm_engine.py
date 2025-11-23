# backend/llm_engine.py
import requests
import os
from dotenv import load_dotenv
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from typing import Optional

load_dotenv()

class ESportsLLM:
    def __init__(self, use_local_model=True, model_path=None):
        """
        Initialize with your trained model OR fallback to Groq API
        
        Args:
            use_local_model: Try to use local trained model first
            model_path: Path to your trained LoRA model
        """
        self.local_model = None
        self.local_tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.using_local = False
        
        # Try to load local model first
        if use_local_model:
            try:
                self._load_local_model(model_path)
            except Exception as e:
                print(f"⚠️  Could not load local model: {e}")
                print("📡 Falling back to Groq API...")
        
        # Setup Groq API as fallback
        if not self.using_local:
            self._setup_groq_api()
    
    def _load_local_model(self, model_path=None):
        """Load your trained model"""
        print("🔧 Loading your trained model...")
        
        # Default paths to check
        possible_paths = [
            model_path,
            "./models/esports-gpu-fast_final",
            "./models/esports-m4-fast_final",
            "../models/esports-gpu-fast_final",
            os.path.expanduser("~/esports-strategy-ai/models/esports-gpu-fast_final")
        ]
        
        model_path = None
        for path in possible_paths:
            if path and os.path.exists(path):
                model_path = path
                break
        
        if not model_path:
            raise ValueError("Trained model not found. Please specify model_path.")
        
        print(f"📂 Loading from: {model_path}")
        
        # Load base model (TinyLlama by default)
        base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        
        # Check if model config specifies different base
        config_path = os.path.join(model_path, "training_info.json")
        if os.path.exists(config_path):
            import json
            with open(config_path) as f:
                info = json.load(f)
                base_model_name = info.get("model", base_model_name)
        
        print(f"📦 Loading base model: {base_model_name}")
        
        # Load tokenizer
        self.local_tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True
        )
        
        if self.local_tokenizer.pad_token is None:
            self.local_tokenizer.pad_token = self.local_tokenizer.eos_token
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            low_cpu_mem_usage=True
        )
        
        # Load LoRA adapter
        print(f"🔧 Loading LoRA adapter from {model_path}")
        self.local_model = PeftModel.from_pretrained(
            base_model,
            model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        
        # Set to eval mode
        self.local_model.eval()
        
        if self.device == "cpu":
            self.local_model = self.local_model.to(self.device)
        
        self.using_local = True
        
        print(f"✅ Local model loaded successfully!")
        print(f"   Device: {self.device.upper()}")
        print(f"   Base: {base_model_name}")
        print(f"   LoRA: {model_path}")
        
        if self.device == "cuda":
            print(f"   GPU Memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    
    def _setup_groq_api(self):
        """Setup Groq API as fallback"""
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
        
        # Current available models
        self.models = {
            "llama-3.1-70b": "llama-3.1-70b-versatile",
            "llama-3.1-8b": "llama-3.1-8b-instant",
            "llama-3.2-3b": "llama-3.2-3b-preview",
            "mixtral": "mixtral-8x7b-32768",
            "gemma2-9b": "gemma2-9b-it"
        }
        
        self.current_model = self.models["llama-3.1-8b"]
        
        print("✅ Groq API configured (fallback)")
        print(f"📦 Model: {self.current_model}")
        
        self._test_connection()
    
    def _test_connection(self):
        """Test Groq API connection"""
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
                print("✅ Groq API connection successful")
                return True
            else:
                print(f"⚠️  API test failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"⚠️  Connection test error: {e}")
            return False
    
    def _generate_local(self, prompt, max_length=1024, temperature=0.7):
        """Generate response using your trained model"""
        try:
            # Format prompt
            formatted_prompt = f"{prompt}\n\nResponse:"
            
            # Tokenize
            inputs = self.local_tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.local_model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.local_tokenizer.pad_token_id,
                    eos_token_id=self.local_tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=3
                )
            
            # Decode
            full_response = self.local_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the response part (after the prompt)
            if "Response:" in full_response:
                response = full_response.split("Response:", 1)[1].strip()
            else:
                response = full_response[len(formatted_prompt):].strip()
            
            return response
            
        except Exception as e:
            print(f"❌ Local generation error: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _generate_groq(self, prompt, max_length=1024, temperature=0.7):
        """Generate response using Groq API"""
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
        
        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                text = result["choices"][0]["message"]["content"]
                return text
            else:
                error_data = response.json()
                error_msg = error_data.get("error", {}).get("message", "Unknown error")
                return f"Error: {error_msg}"
                
        except Exception as e:
            print(f"❌ Groq API error: {e}")
            raise
    
    def generate_response(self, prompt, max_length=1024, temperature=0.7, retry_count=3):
        """
        Generate response using trained model or Groq API fallback
        """
        for attempt in range(retry_count):
            try:
                print(f"🔄 Generating response (attempt {attempt + 1}/{retry_count})...")
                start_time = time.time()
                
                # Try local model first
                if self.using_local:
                    try:
                        response = self._generate_local(prompt, max_length, temperature)
                        elapsed_time = time.time() - start_time
                        print(f"✅ Local model response in {elapsed_time:.2f}s")
                        return response
                    except Exception as e:
                        print(f"⚠️  Local model failed: {e}")
                        if attempt < retry_count - 1:
                            print("🔄 Retrying with Groq API...")
                            # Temporarily disable local for this request
                            temp_using_local = self.using_local
                            self.using_local = False
                            continue
                        raise
                
                # Use Groq API
                response = self._generate_groq(prompt, max_length, temperature)
                elapsed_time = time.time() - start_time
                print(f"✅ Groq API response in {elapsed_time:.2f}s")
                return response
                
            except Exception as e:
                if attempt < retry_count - 1:
                    print(f"⚠️  Attempt {attempt + 1} failed: {e}")
                    time.sleep(3)
                    continue
                else:
                    print(f"❌ All attempts failed: {e}")
                    return f"Error: Failed to generate response after {retry_count} attempts."
        
        return "Error: Failed to generate response."
    
    def get_model_info(self):
        """Get information about current model"""
        if self.using_local:
            return {
                "type": "local_trained",
                "device": self.device,
                "model": "Custom LoRA Fine-tuned Model",
                "base": "TinyLlama-1.1B"
            }
        else:
            return {
                "type": "groq_api",
                "model": self.current_model
            }
    
    def switch_model(self, model_name):
        """Switch Groq model (only works if using API)"""
        if self.using_local:
            print("⚠️  Currently using local model. Cannot switch Groq models.")
            return
        
        if model_name in self.models:
            self.current_model = self.models[model_name]
            print(f"✅ Switched to model: {self.current_model}")
        else:
            print(f"⚠️  Unknown model: {model_name}")
            print(f"Available: {list(self.models.keys())}")