#!/usr/bin/env python3
"""
Test script for ESports Strategy AI API
Run this after starting the server to verify everything works
"""

import requests
import json

API_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("\n🔍 Testing /health...")
    response = requests.get(f"{API_URL}/health")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    return response.status_code == 200

def test_prompt_options():
    """Test prompt options endpoint"""
    print("\n🔍 Testing /prompt-options...")
    response = requests.get(f"{API_URL}/prompt-options")
    print(f"   Status: {response.status_code}")
    data = response.json()
    print(f"   Length options: {data.get('length_options')}")
    print(f"   Style options: {data.get('style_options')}")
    return response.status_code == 200

def test_analyze():
    """Test main analysis endpoint"""
    print("\n🔍 Testing /analyze...")
    
    # Load sample match
    with open("sample_matches/sample_match.json", "r") as f:
        match_data = json.load(f)
    
    payload = {
        "match_data": match_data,
        "focus_team": "team_a",
        "prompt_controls": {
            "length": "short",
            "focus": ["overall"],
            "style": "analytical"
        }
    }
    
    response = requests.post(f"{API_URL}/analyze", json=payload)
    print(f"   Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        if "error" in data:
            print(f"   ❌ Error: {data['error']}")
            return False
        result = data.get("result", "")
        print(f"   ✅ Response length: {len(result)} chars")
        print(f"   Preview: {result[:200]}...")
        return True
    return False

def test_quick_summary():
    """Test quick summary endpoint"""
    print("\n🔍 Testing /quick-summary...")
    
    with open("sample_matches/sample_match.json", "r") as f:
        match_data = json.load(f)
    
    payload = {"match_data": match_data}
    
    response = requests.post(f"{API_URL}/quick-summary", json=payload)
    print(f"   Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        result = data.get("result", "")
        print(f"   ✅ Response length: {len(result)} chars")
        return True
    return False

def main():
    print("=" * 60)
    print("   ESports Strategy AI - API Tests")
    print("=" * 60)
    
    results = {
        "Health": test_health(),
        "Prompt Options": test_prompt_options(),
        "Analyze": test_analyze(),
        "Quick Summary": test_quick_summary(),
    }
    
    print("\n" + "=" * 60)
    print("   Results Summary")
    print("=" * 60)
    
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {name}: {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    print(f"\n   Total: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n🎉 All tests passed! The API is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the server logs.")

if __name__ == "__main__":
    main()
