# ESports Strategy Summarizer - API Examples

## Example 1: Match Summarization (Analytical Style)

### Request
```bash
curl -X POST http://localhost:5000/api/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "match_data": {
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
        "42:15 - Team Liquid destroys ancient"
      ],
      "commentary": "An intense match between two top-tier teams. Team Liquid'\''s Invoker controlled the mid game with exceptional spell combos.",
      "statistics": {
        "Team Liquid Kills": "45",
        "Evil Geniuses Kills": "32",
        "Team Liquid Gold": "65.2k",
        "Evil Geniuses Gold": "58.1k"
      }
    },
    "style": "analytical",
    "focus": "team_fights",
    "length": "medium"
  }'
```

### Response
```json
{
  "success": true,
  "summary": "Team Liquid demonstrated superior team fighting execution throughout this 42-minute match. The key turning point came at 32 minutes when they achieved a clean team wipe near the Roshan pit, building on their earlier 4-1 victory at 18:45. Despite Evil Geniuses securing Roshan at 25:30, Team Liquid's Invoker and Earthshaker combination proved decisive in controlling major engagements. The final statistics (45-32 kills, 7k gold advantage) reflect Team Liquid's consistent ability to win crucial team fights.",
  "metadata": {
    "style": "analytical",
    "focus": "team_fights",
    "length": "medium",
    "temperature": 0.7,
    "timestamp": "2024-01-15T14:30:00.123456"
  }
}
```

---

## Example 2: Match Summarization (Narrative Style)

### Request
```bash
curl -X POST http://localhost:5000/api/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "match_data": {
      "team_a": "OG",
      "team_b": "Team Secret",
      "winner": "OG",
      "duration": "51:34",
      "events": [
        "10:00 - Team Secret dominates early game, 8-1 kill lead",
        "22:00 - OG stages comeback with 5-0 team wipe",
        "35:00 - OG secures mega creeps",
        "51:34 - OG wins after epic comeback"
      ]
    },
    "style": "narrative",
    "focus": "macro",
    "length": "long"
  }'
```

### Response
```json
{
  "success": true,
  "summary": "This match will be remembered as one of the greatest comebacks in professional Dota 2. The story began with Team Secret writing what looked like a perfect script, dominating the first 20 minutes with an 8-1 kill advantage. But OG, true to their reputation, refused to give up. The turning point came at 22 minutes—a perfectly executed team fight that resulted in a complete team wipe. From that moment, momentum shifted dramatically. OG methodically converted their newfound advantage, eventually securing mega creeps at 35 minutes. The final 15 minutes saw Team Secret fighting valiantly but ultimately falling to OG's relentless pressure. When the ancient fell at 51:34, it marked not just a victory, but a masterclass in patience and precision under pressure.",
  "metadata": {
    "style": "narrative",
    "focus": "macro",
    "length": "long",
    "temperature": 0.7,
    "timestamp": "2024-01-15T14:32:15.654321"
  }
}
```

---

## Example 3: Tactical Recommendations

### Request
```bash
curl -X POST http://localhost:5000/api/recommendations \
  -H "Content-Type: application/json" \
  -d '{
    "match_data": {
      "team_a": "Fnatic",
      "team_b": "PSG.LGD",
      "winner": "PSG.LGD",
      "duration": "38:42",
      "team_a_composition": "Spectre, Shadow Fiend, Disruptor, Bounty Hunter, Oracle",
      "team_b_composition": "Faceless Void, Dragon Knight, Magnus, Grimstroke, Vengeful Spirit",
      "events": [
        "Early game: Spectre struggled to farm safely",
        "12:00 - Multiple failed ganks by Bounty Hunter",
        "20:00 - PSG.LGD secured all objectives",
        "30:00 - Spectre still underfarmed, game lost"
      ],
      "statistics": {
        "Fnatic Kills": "18",
        "PSG.LGD Kills": "35",
        "Fnatic Gold": "48.2k",
        "PSG.LGD Gold": "67.8k"
      }
    },
    "team": "team_a",
    "recommendation_depth": 4
  }'
```

### Response
```json
{
  "success": true,
  "recommendations": "1. DRAFT ADJUSTMENTS: Consider picking tankier position 3 heroes (e.g., Axe, Centaur, Bristleback) to create space for Spectre. The current lineup lacks frontline presence to protect your late-game carry during crucial farming phases.\n\n2. EARLY GAME STRATEGY: Abandon the aggressive Bounty Hunter roaming strategy against lineups with strong counter-initiation like Magnus. Instead, focus on defensive warding and protecting Spectre's farm until she reaches critical item timings (Radiance + Manta).\n\n3. OBJECTIVE PRIORITIZATION: When behind, avoid contesting every objective. The multiple failed attempts to stop PSG.LGD's objectives led to further gold deficit. Instead, trade objectives—give up outer towers while securing your own farm and stacking camps for Spectre.\n\n4. TIMING WINDOWS: Recognize that Faceless Void + Magnus peaks around 25-30 minutes. Plan your power spike around 35+ minutes when Spectre becomes truly online. Play defensively until this timing, using Disruptor to punish overextensions rather than forcing fights you're not ready for.",
  "metadata": {
    "team": "team_a",
    "recommendation_depth": 4,
    "timestamp": "2024-01-15T14:35:42.987654"
  }
}
```

---

## Example 4: Turning Points Identification

### Request
```bash
curl -X POST http://localhost:5000/api/turning-points \
  -H "Content-Type: application/json" \
  -d '{
    "match_data": {
      "team_a": "T1",
      "team_b": "Gen.G",
      "winner": "T1",
      "events": [
        "15:00 - Gen.G takes Baron, gains 8k gold lead",
        "23:00 - T1 steals Elder Dragon with perfect smite",
        "28:00 - T1 wins decisive team fight at mid lane",
        "32:00 - T1 secures Baron and pushes to win"
      ],
      "commentary": "What a turnaround! T1 was on the brink of defeat but that Elder Dragon steal completely changed the game."
    }
  }'
```

### Response
```json
{
  "success": true,
  "turning_points": "1. ELDER DRAGON STEAL (23:00) - The pivotal moment\n   What happened: T1's jungler executed a perfect 50/50 smite to steal Elder Dragon from Gen.G\n   Why significant: This denied Gen.G their win condition while giving T1 the Elder buff, effectively neutralizing the 8k gold disadvantage\n   Impact: Shifted momentum completely and gave T1 the confidence to take fights they were previously avoiding\n\n2. MID LANE TEAM FIGHT (28:00) - The momentum converter\n   What happened: With Elder Dragon buff active, T1 engaged Gen.G at mid lane and won the team fight\n   Why significant: This was T1's first major team fight victory and came at a crucial moment when Gen.G was setting up for their second Baron\n   Impact: Eliminated Gen.G's gold lead and map control, opening up the map for T1's vision control\n\n3. BARON SECURE AND FINAL PUSH (32:00) - The closer\n   What happened: T1 secured Baron uncontested and used the buff to end the game\n   Why significant: This Baron was the first objective T1 secured cleanly all game, demonstrating complete map control reversal\n   Impact: Enabled T1 to break Gen.G's base and secure victory in what seemed like an impossible comeback",
  "metadata": {
    "timestamp": "2024-01-15T14:38:21.147258"
  }
}
```

---

## Example 5: Complete Match Analysis

### Request
```bash
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "match_data": {
      "team_a": "Cloud9",
      "team_b": "TSM",
      "winner": "Cloud9",
      "duration": "35:17",
      "team_a_composition": "Gangplank, Graves, Orianna, Jinx, Thresh",
      "team_b_composition": "Ornn, Lee Sin, Syndra, Kai'\''Sa, Nautilus",
      "events": [
        "8:00 - Cloud9 secures first blood and first dragon",
        "14:00 - TSM takes first tower",
        "21:00 - Cloud9 wins Baron fight 4-1",
        "28:00 - Cloud9 takes down inhibitor",
        "35:17 - Cloud9 wins"
      ],
      "statistics": {
        "Cloud9 Kills": "22",
        "TSM Kills": "14",
        "Cloud9 Gold": "65.3k",
        "TSM Gold": "58.9k",
        "Cloud9 Dragons": "3",
        "TSM Dragons": "1"
      }
    },
    "style": "tactical",
    "focus": "objectives",
    "team": "team_a",
    "recommendation_depth": 3
  }'
```

### Response
```json
{
  "success": true,
  "analysis": {
    "summary": "Cloud9 executed a textbook objective-focused victory against TSM in 35 minutes. The match demonstrated superior macro play, with Cloud9 securing 3 dragons compared to TSM's 1. The crucial Baron fight at 21 minutes (4-1 in Cloud9's favor) exemplified their superior objective control and team fight positioning. Cloud9's composition (Gangplank, Graves, Orianna, Jinx, Thresh) provided excellent objective taking speed and zone control, particularly effective around Baron pit. The 6.4k gold lead reflected consistent pressure on neutral objectives rather than pure kill advantage (22-14 kills).",
    "recommendations": "1. OBJECTIVE PRIORITY SEQUENCING: Continue prioritizing early dragon control as you did in this game. Your 3-dragon stack gave you significant combat stats advantage. In future games, consider timing dragon spawns with your bot lane recall timings to maximize efficiency.\n\n2. BARON SETUP AND CONTROL: Your Baron fight execution at 21 minutes was excellent. To improve further, establish deeper vision control 1-2 minutes before Baron spawn. Use Graves and Thresh to zone enemy jungler from approach paths.\n\n3. SIEGE OPTIMIZATION: Your inhibitor take at 28 minutes could have been faster. With Jinx's range and Gangplank's barrels, you can poke down towers from safer distances. Practice spacing around sieges to avoid unnecessary casualties and maximize structure damage.",
    "turning_points": "1. EARLY DRAGON SECURE (8:00)\n   What happened: Cloud9 secured first blood and immediately rotated to take first dragon\n   Significance: Set the tempo for the entire game, establishing Cloud9's objective priority mindset\n   Impact: Forced TSM into reactive defensive positions for the rest of early game\n\n2. BARON FIGHT VICTORY (21:00)\n   What happened: 4-1 team fight victory at Baron pit for Cloud9\n   Significance: This was the match-defining fight that gave Cloud9 both Baron buff and map control\n   Impact: Enabled Cloud9 to break into TSM's base and take inhibitor within 7 minutes\n\n3. INHIBITOR BREACH (28:00)\n   What happened: Cloud9 took down middle inhibitor with Baron buff\n   Significance: With super minions and objective control, TSM could no longer contest neutrals\n   Impact: Sealed the game outcome as TSM was forced to defend base, allowing Cloud9 to secure soul and close out"
  },
  "metadata": {
    "timestamp": "2024-01-15T14:42:07.369258",
    "match": {
      "team_a": "Cloud9",
      "team_b": "TSM",
      "winner": "Cloud9"
    }
  }
}
```

---

## Example 6: Using Python Requests Library

```python
import requests
import json

# Configuration
API_URL = "http://localhost:5000"

# Match data
match = {
    "team_a": "Team Liquid",
    "team_b": "Natus Vincere",
    "winner": "Team Liquid",
    "duration": "44:32",
    "events": [
        "10:00 - First blood to Team Liquid",
        "25:00 - Team fight won by Team Liquid (5-2)",
        "40:00 - Roshan secured by Team Liquid",
        "44:32 - Team Liquid wins"
    ]
}

# Get summary
response = requests.post(
    f"{API_URL}/api/summarize",
    json={
        "match_data": match,
        "style": "analytical",
        "focus": "team_fights",
        "length": "medium"
    }
)

if response.status_code == 200:
    result = response.json()
    print("Summary:", result["summary"])
else:
    print("Error:", response.json())

# Get recommendations
response = requests.post(
    f"{API_URL}/api/recommendations",
    json={
        "match_data": match,
        "team": "team_b",  # Get recommendations for the losing team
        "recommendation_depth": 3
    }
)

if response.status_code == 200:
    result = response.json()
    print("\nRecommendations:", result["recommendations"])
```

---

## Example 7: Using JavaScript Fetch API

```javascript
// Match data
const matchData = {
  team_a: "Fnatic",
  team_b: "NRG",
  winner: "Fnatic",
  duration: "38:15",
  events: [
    "12:00 - Fnatic secures first Baron",
    "25:00 - Major team fight, Fnatic wins 4-1",
    "38:15 - Fnatic wins the match"
  ],
  statistics: {
    "Fnatic Kills": "32",
    "NRG Kills": "21"
  }
};

// Get complete analysis
fetch('http://localhost:5000/api/analyze', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    match_data: matchData,
    style: 'narrative',
    focus: 'objectives',
    team: 'team_a',
    recommendation_depth: 3
  })
})
.then(response => response.json())
.then(data => {
  console.log('Summary:', data.analysis.summary);
  console.log('Recommendations:', data.analysis.recommendations);
  console.log('Turning Points:', data.analysis.turning_points);
})
.catch(error => console.error('Error:', error));
```

---

## Tips for Best Results

1. **Provide detailed events**: More context leads to better analysis
2. **Include commentary**: Adds narrative depth to summaries
3. **Add statistics**: Helps model understand match dynamics
4. **Choose appropriate style**:
   - `analytical`: For coaches and analysts
   - `narrative`: For content creators and storytelling
   - `bullet`: For quick reviews
   - `tactical`: For strategic deep-dives

4. **Adjust temperature**:
   - Lower (0.3-0.5): More focused and deterministic
   - Medium (0.6-0.8): Balanced creativity and consistency
   - Higher (0.9-1.2): More creative and varied outputs
