"""
Test Client for ESports Strategy Summarizer API
Demonstrates how to use the backend API endpoints
"""

import requests
import json
from typing import Dict

# API Configuration
API_BASE_URL = "http://localhost:8080"

def test_health():
    """Test health endpoint"""
    print("\n" + "="*60)
    print("Testing Health Endpoint")
    print("="*60)
    
    response = requests.get(f"{API_BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

def test_config():
    """Test config endpoint"""
    print("\n" + "="*60)
    print("Getting Configuration Options")
    print("="*60)
    
    response = requests.get(f"{API_BASE_URL}/api/config")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

def test_summarize():
    """Test match summarization"""
    print("\n" + "="*60)
    print("Testing Match Summarization")
    print("="*60)
    
    # Sample match data
    match_data = {
        "team_a": "Team Liquid",
        "team_b": "Evil Geniuses",
        "winner": "Team Liquid",
        "duration": "42:15",
        "team_a_composition": "Invoker, Anti-Mage, Lion, Earthshaker, Crystal Maiden",
        "team_b_composition": "Phantom Assassin, Queen of Pain, Rubick, Tidehunter, Ancient Apparition",
        "events": [
            "5:30 - Team Liquid secures first blood mid lane",
            "12:00 - Evil Geniuses takes first tower bot lane",
            "18:45 - Major team fight at Roshan pit, Team Liquid wins 4-1",
            "25:30 - Evil Geniuses secures Roshan",
            "32:00 - Team Liquid wins decisive team fight, wipes Evil Geniuses",
            "35:15 - Team Liquid takes mid barracks",
            "42:15 - Team Liquid destroys ancient, wins the match"
        ],
        "commentary": "An intense match between two top-tier teams. Team Liquid's Invoker controlled the mid game with exceptional spell combos, while Evil Geniuses struggled to protect their Phantom Assassin. The turning point came at 32 minutes when Team Liquid caught Evil Geniuses out of position near the Roshan pit.",
        "statistics": {
            "Team Liquid Kills": "45",
            "Evil Geniuses Kills": "32",
            "Team Liquid Gold": "65.2k",
            "Evil Geniuses Gold": "58.1k",
            "Team Liquid Towers": "11",
            "Evil Geniuses Towers": "7"
        }
    }
    
    payload = {
        "match_data": match_data,
        "style": "analytical",
        "focus": "team_fights",
        "length": "medium",
        "temperature": 0.7
    }
    
    response = requests.post(
        f"{API_BASE_URL}/api/summarize",
        json=payload
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"\nSummary:\n{result['summary']}\n")
        print(f"Metadata: {json.dumps(result['metadata'], indent=2)}")
    else:
        print(f"Error: {response.text}")

def test_recommendations():
    """Test tactical recommendations"""
    print("\n" + "="*60)
    print("Testing Tactical Recommendations")
    print("="*60)
    
    match_data = {
        "team_a": "Team Liquid",
        "team_b": "Evil Geniuses",
        "winner": "Evil Geniuses",
        "duration": "38:42",
        "team_a_composition": "Spectre, Shadow Fiend, Disruptor, Bounty Hunter, Oracle",
        "events": [
            "Early game dominated by Evil Geniuses",
            "Team Liquid struggled to protect Spectre",
            "Multiple failed ganks by Bounty Hunter",
            "Evil Geniuses secured all Roshan kills",
            "Team Liquid lost all tier 2 towers before 25 minutes"
        ],
        "statistics": {
            "Team Liquid Kills": "28",
            "Evil Geniuses Kills": "41",
            "Team Liquid Gold": "52.3k",
            "Evil Geniuses Gold": "68.7k"
        }
    }
    
    payload = {
        "match_data": match_data,
        "team": "team_a",
        "recommendation_depth": 4,
        "temperature": 0.7
    }
    
    response = requests.post(
        f"{API_BASE_URL}/api/recommendations",
        json=payload
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"\nRecommendations:\n{result['recommendations']}\n")
        print(f"Metadata: {json.dumps(result['metadata'], indent=2)}")
    else:
        print(f"Error: {response.text}")

def test_turning_points():
    """Test turning points identification"""
    print("\n" + "="*60)
    print("Testing Turning Points Identification")
    print("="*60)
    
    match_data = {
        "team_a": "OG",
        "team_b": "Team Secret",
        "winner": "OG",
        "events": [
            "10:00 - Team Secret takes early lead with 5-1 kill score",
            "15:30 - OG loses mid tower, down 8k gold",
            "22:00 - OG wins major team fight 5-0, momentum shifts",
            "28:15 - OG secures Roshan and Aegis",
            "35:00 - Another team fight win for OG, takes megas",
            "39:45 - OG wins the game despite early deficit"
        ],
        "commentary": "What an incredible comeback! OG was down 10k gold at 20 minutes but managed to turn it around with superior team fighting. The key moment was the 5-0 team wipe at 22 minutes that gave them the confidence to take over the game."
    }
    
    payload = {
        "match_data": match_data,
        "temperature": 0.7
    }
    
    response = requests.post(
        f"{API_BASE_URL}/api/turning-points",
        json=payload
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"\nTurning Points:\n{result['turning_points']}\n")
    else:
        print(f"Error: {response.text}")

def test_full_analysis():
    """Test complete match analysis"""
    print("\n" + "="*60)
    print("Testing Complete Match Analysis")
    print("="*60)
    
    match_data = {
        "team_a": "Fnatic",
        "team_b": "PSG.LGD",
        "winner": "Fnatic",
        "duration": "51:23",
        "team_a_composition": "Morphling, Templar Assassin, Enigma, Bane, Snapfire",
        "team_b_composition": "Faceless Void, Dragon Knight, Magnus, Grimstroke, Vengeful Spirit",
        "events": [
            "8:00 - PSG.LGD gets first blood",
            "14:30 - Fnatic secures first tower",
            "21:00 - Major team fight, PSG.LGD wins 4-2",
            "29:45 - Fnatic steals Roshan",
            "38:00 - Game-deciding team fight, Fnatic wins 5-1",
            "45:15 - Fnatic takes mega creeps",
            "51:23 - Fnatic wins"
        ],
        "statistics": {
            "Fnatic Kills": "52",
            "PSG.LGD Kills": "48",
            "Fnatic Gold": "89.4k",
            "PSG.LGD Gold": "84.2k"
        }
    }
    
    payload = {
        "match_data": match_data,
        "style": "tactical",
        "focus": "objectives",
        "team": "team_a",
        "recommendation_depth": 3
    }
    
    response = requests.post(
        f"{API_BASE_URL}/api/analyze",
        json=payload
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        analysis = result['analysis']
        
        print("\n--- SUMMARY ---")
        print(analysis['summary'])
        
        print("\n--- RECOMMENDATIONS ---")
        print(analysis['recommendations'])
        
        print("\n--- TURNING POINTS ---")
        print(analysis['turning_points'])
        
        print(f"\nMetadata: {json.dumps(result['metadata'], indent=2)}")
    else:
        print(f"Error: {response.text}")

def main():
    """Run all tests"""
    print("="*60)
    print("ESports Strategy Summarizer - API Test Client")
    print("="*60)
    
    try:
        # Basic checks
        test_health()
        test_config()
        
        # Main functionality tests
        test_summarize()
        test_recommendations()
        test_turning_points()
        test_full_analysis()
        
        print("\n" + "="*60)
        print("All tests completed!")
        print("="*60)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to API server")
        print("   Make sure the Flask backend is running on http://localhost:8080")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()