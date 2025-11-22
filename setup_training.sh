#!/bin/bash

# ============================================================
# ESports Strategy AI - Model Training Setup Script
# ============================================================
# This script sets up everything needed to train/fine-tune
# a model for esports match analysis
# ============================================================

set -e

echo "============================================================"
echo "    ESports Strategy AI - Training Setup"
echo "============================================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$SCRIPT_DIR"

echo -e "${BLUE}📁 Project directory: $PROJECT_DIR${NC}"

# ============================================================
# Create directories
# ============================================================
echo -e "${YELLOW}📂 Creating directories...${NC}"

mkdir -p "$PROJECT_DIR/training"
mkdir -p "$PROJECT_DIR/training/data"
mkdir -p "$PROJECT_DIR/training/models"
mkdir -p "$PROJECT_DIR/training/logs"
mkdir -p "$PROJECT_DIR/training/checkpoints"

echo -e "${GREEN}✓ Directories created${NC}"

# ============================================================
# Create training requirements
# ============================================================
echo -e "${YELLOW}📦 Creating training requirements...${NC}"

cat > "$PROJECT_DIR/training/requirements_training.txt" << 'EOF'
# Training Requirements
# Requires CUDA-capable GPU for efficient training

# Core ML
torch>=2.0.0
transformers>=4.36.0
datasets>=2.16.0
accelerate>=0.25.0

# Efficient Fine-tuning
peft>=0.7.0
bitsandbytes>=0.41.0
trl>=0.7.0

# Data Processing
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0

# Evaluation
rouge-score>=0.1.2
nltk>=3.8.1
evaluate>=0.4.0

# Utilities
python-dotenv>=1.0.0
tqdm>=4.66.0
wandb>=0.16.0
tensorboard>=2.15.0

# Jupyter (optional)
jupyter>=1.0.0
ipywidgets>=8.1.0
EOF

echo -e "${GREEN}✓ requirements_training.txt created${NC}"

# ============================================================
# Create Data Preparation Script
# ============================================================
echo -e "${YELLOW}📊 Creating data preparation script...${NC}"

cat > "$PROJECT_DIR/training/prepare_data.py" << 'PREPARE_DATA_EOF'
"""
Data Preparation Script for ESports Strategy AI
Downloads and prepares datasets for training:
1. DialogSum - for dialogue summarization practice
2. Custom ESports data - for game-specific training

Usage:
    python prepare_data.py
    python prepare_data.py --only-esports  # Skip DialogSum
    python prepare_data.py --samples 1000  # Limit DialogSum samples
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from tqdm import tqdm


class ESportsDataPreparation:
    """Prepares training data for esports strategy model."""
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def download_dialogsum(self, max_samples: int = 1000) -> Dataset:
        """
        Download DialogSum dataset from HuggingFace.
        Good for learning general summarization skills.
        """
        print("📥 Downloading DialogSum dataset...")
        
        try:
            dataset = load_dataset("knkarthick/dialogsum")
            print(f"   ✓ Downloaded {len(dataset['train'])} training samples")
            
            # Convert to instruction format
            training_data = []
            for item in tqdm(dataset['train'][:max_samples], desc="Converting DialogSum"):
                training_data.append({
                    "instruction": "Summarize the following dialogue concisely, capturing the main points and outcome.",
                    "input": item['dialogue'],
                    "output": item['summary']
                })
            
            ds = Dataset.from_list(training_data)
            ds.save_to_disk(self.data_dir / "dialogsum_processed")
            print(f"   ✓ Saved {len(ds)} samples to {self.data_dir / 'dialogsum_processed'}")
            
            return ds
            
        except Exception as e:
            print(f"   ❌ Error downloading DialogSum: {e}")
            return None
    
    def create_esports_dataset(self) -> Dataset:
        """
        Create comprehensive esports training dataset.
        Includes various types of analysis tasks.
        """
        print("🎮 Creating ESports training dataset...")
        
        training_samples = []
        
        # ============================================================
        # MATCH SUMMARIES - Different styles and games
        # ============================================================
        
        match_summary_samples = [
            # League of Legends - Comeback victory
            {
                "instruction": "Provide an analytical summary of this League of Legends match, identifying the key factors that determined the outcome.",
                "input": """Match: Cloud9 vs Team Liquid (LCS Finals)
Duration: 35 minutes
Winner: Cloud9

Teams:
- Cloud9 (Blue): Gnar, Lee Sin, Azir, Jinx, Thresh
- Team Liquid (Red): Renekton, Viego, Viktor, Kai'Sa, Nautilus

Key Events:
- [5:15] First Blood: Liquid's Viego ganks mid, kills Azir
- [7:00] Dragon: Liquid takes Mountain Drake uncontested
- [10:30] Tower: Liquid takes first tower mid
- [15:00] Gold Lead: Liquid ahead by 1500 gold
- [18:00] Dragon Fight: Azir shuffle catches 3, C9 wins 4-1
- [25:30] Baron: C9 wins fight 3-0, secures Baron
- [32:00] Dragon Soul: C9 secures Infernal Soul
- [35:00] Game End: Jinx pentakill, C9 wins

Stats: Kills 22-15, Towers 9-3, Dragons 4-2""",
                "output": """## Match Summary

Cloud9 secured a dramatic comeback victory against Team Liquid in a 35-minute Game 1 of the LCS Finals, overcoming a significant early deficit through superior teamfight execution.

### Early Game (0-15 min)
Team Liquid executed their early-game composition flawlessly, securing first blood through a Viego gank at 5:15 and building a 1500 gold lead by 15 minutes. Their Renekton-Viego duo created constant pressure, taking first tower and Mountain Drake uncontested.

### The Turning Point (18:00)
The pivotal moment came at the Infernal Drake fight. Despite being behind, Cloud9's Azir delivered a game-changing shuffle, catching three Liquid members out of position. This resulted in a decisive 4-1 teamfight victory that completely shifted momentum.

### Mid-Late Game (18-35 min)
Following the dragon fight, Cloud9's scaling composition came online. At 25 minutes, a crucial Thresh hook on Viego led to a 3-0 fight and Baron secure. With Baron buff, C9 accumulated dragons and eventually secured Infernal Soul at 32 minutes.

### Conclusion
The game ended with a spectacular Jinx pentakill, capping off C9's comeback with a 22-15 kill score. This match demonstrated that early leads are meaningless without the ability to teamfight against scaling compositions.

**Key Factors:**
1. Azir's mechanical outplay at dragon
2. Thresh's pick potential in mid-game
3. Jinx's scaling overwhelming Liquid's early-game comp"""
            },
            
            # Dota 2 - Timing window analysis
            {
                "instruction": "Analyze this Dota 2 match focusing on timing windows and strategic execution.",
                "input": """Match: Phoenix Esports vs Dragon Gaming (TI Semifinals)
Duration: 42 minutes
Winner: Dragon Gaming (Dire)

Compositions:
- Phoenix (Radiant): Luna, Puck, Mars, Chen, Lich
- Dragon (Dire): Anti-Mage, Storm Spirit, Beastmaster, Rubick, Vengeful Spirit

Key Events:
- [8:00] Tower: Phoenix takes T1 mid with Chen push
- [12:00] Fight: Phoenix wins 4v4, takes T1 top
- [15:00] Networth: Phoenix leads by 8000 gold
- [18:00] High Ground: Phoenix attempts push, loses 3 to Roar initiation
- [22:00] Farm: AM completes Battle Fury, Storm has Orchid
- [25:00] Roshan: Dragon takes Roshan, Aegis to AM
- [32:00] Roshan: Second Roshan, Dragon fully online
- [38:00] Fight: Dragon wins decisive fight, pushes mid
- [42:00] GG: Dragon destroys ancient""",
                "output": """## Strategic Analysis: Timing Windows in Dota 2

This match exemplifies the critical importance of executing timing windows in Dota 2. Phoenix Esports' early-game dominance meant nothing once their window closed.

### Phoenix's Game Plan
Phoenix drafted a classic timing push lineup with Luna-Chen deathball potential. Their win condition was clear: end the game before 25-30 minutes when Anti-Mage and Storm Spirit would come online.

### Early Execution (0-15 min)
Phoenix executed perfectly in the early game:
- T1 mid at 8 minutes with Chen creeps
- Won the 4v4 at 12 minutes
- Built an 8000 gold lead by 15 minutes

Everything was going according to plan.

### The Critical Failure (18:00)
The high ground attempt at 18 minutes was the game-deciding moment. Three critical errors:
1. **No detection** for Beastmaster's potential initiation
2. **Luna positioning** too aggressive for Roar range
3. **No buyback consideration** for this crucial timing

Losing 3 heroes without taking a single tower gave Dragon exactly what they needed: time.

### Dragon's Recovery (18-32 min)
With the failed push, Dragon's cores had space:
- Anti-Mage: Battle Fury at 22 minutes (late but acceptable given pressure)
- Storm Spirit: Orchid at 22 minutes
- First Roshan at 25 minutes provided insurance

By 32 minutes, Dragon was fully online with second Aegis.

### Late Game Inevitability (32-42 min)
Phoenix's composition had no answer to a 6-slotted Anti-Mage with Storm Spirit backup. The 38-minute fight was one-sided, and the game ended at 42 minutes.

### Key Takeaways
1. **Timing compositions must execute** - There's no plan B
2. **High ground requires preparation** - Smoke, detection, item timings
3. **One failed push can lose the game** against hard carries
4. **Dragon showed excellent discipline** - Didn't panic, trusted their draft"""
            },
            
            # Overwatch 2 - Adaptation analysis
            {
                "instruction": "Analyze the team adaptation and counter-play in this Overwatch 2 match.",
                "input": """Match: Nexus Gaming vs Titan Esports (OWL Playoffs)
Map: Numbani (Hybrid)
Duration: 8 minutes
Winner: Nexus Gaming (Attack)

Team Compositions:
Initial:
- Nexus (Attack): Winston, D.Va, Tracer, Genji, Ana, Lucio
- Titan (Defense): Sigma, Orisa, Bastion, Soldier, Baptiste, Mercy

After Adaptation:
- Nexus: Winston, D.Va, Sombra, Genji, Ana, Lucio
- Titan: Ball, Tracer, Sojourn, Tracer, Kiriko, Lucio

Key Events:
- [1:30] First Fight: Nexus dives bunker, gets shredded by Bastion (0-4)
- [2:30] Regroup: Nexus swaps Tracer to Sombra
- [3:00] EMP Fight: Sombra EMPs, bunker collapses, Nexus wins 4-0
- [4:00] Point A: Captured with 3 minutes remaining
- [5:00] Titan Swap: Titan goes mobile composition
- [6:30] Checkpoint: Nexus reaches second checkpoint
- [8:00] Final Fight: Titan uses support ults early, Nexus cleans up with Blade""",
                "output": """## Adaptation Analysis: The Art of Mid-Match Adjustments

This Numbani attack showcased elite-level adaptation from Nexus Gaming, turning a terrible first fight into a dominant map victory.

### Initial Composition Clash

**Nexus's Problem:** Their dive composition (Winston/D.Va/Tracer/Genji) is designed to isolate targets and win through mobility. Titan's bunker (Sigma/Orisa/Bastion) completely counters this by:
- Providing massive sustained damage (Bastion)
- Double shield protection
- Baptiste immortality field
- No isolated targets to pick

**First Fight Disaster:** Nexus dove without scouting, lost 4 members to Bastion's sustained damage. Classic dive vs bunker mismatch.

### The Adaptation (2:30)

Nexus made the correct read immediately. The Tracer-to-Sombra swap was perfect because:

1. **EMP counters bunker fundamentally** - Removes all shields and abilities
2. **Hack prevents Bastion transformation** - Neutralizes the core threat
3. **Maintains dive capability** - Sombra still enables aggressive plays

### Execution Excellence (3:00-4:00)

The EMP at 3:00 was textbook:
- Waited for Baptiste to use Immortality
- EMPed from high ground (safe angle)
- Team followed up immediately
- 4-0 fight win, Point A captured

### Titan's Counter-Adaptation (5:00)

Recognizing bunker wouldn't work on Point B's open layout, Titan swapped to mobile/dive mirror. This was correct strategically but:
- They lacked practice on this composition
- Coordination suffered during transition
- Mental damage from Point A collapse

### The Finishing Blow (8:00)

Titan's final fight error was critical: Both support ultimates (likely Sound Barrier + Suzu) used proactively instead of reactively to Genji Blade. This left them defenseless when the Blade came out.

### Key Lessons

1. **Scout before committing** - Nexus's first fight was avoidable
2. **Have counter-picks ready** - Sombra was the perfect answer
3. **Adaptation requires practice** - Titan's swap exposed unfamiliarity
4. **Ultimate economy wins fights** - Save defensive ults for enemy offensive ults"""
            },
        ]
        
        training_samples.extend(match_summary_samples)
        
        # ============================================================
        # TURNING POINTS ANALYSIS
        # ============================================================
        
        turning_point_samples = [
            {
                "instruction": "Identify and analyze the key turning points in this match.",
                "input": """Game: League of Legends
Match: TSM vs 100 Thieves
Duration: 38 minutes
Winner: TSM

Timeline:
- [0-10] 100T dominates lanes, 2k gold lead
- [12:00] 100T takes Herald, crashes top
- [15:00] 100T has 3.5k gold lead, all outer towers
- [18:00] TSM wins fight at dragon 4-2
- [22:00] 100T throws at Baron, TSM steals + ace
- [28:00] TSM takes Baron with numbers advantage
- [32:00] TSM secures Elder Dragon
- [38:00] TSM ends the game

100T comp: Early/mid game focused
TSM comp: Late game scaling""",
                "output": """## Key Turning Points Analysis

### Turning Point #1: Dragon Fight (18:00) - Impact: HIGH

**What Happened:** Despite being down 3.5k gold, TSM won a 4-2 teamfight at dragon.

**Why It Mattered:**
- First sign that 100T's lead wasn't insurmountable
- TSM's scaling composition showed it could teamfight
- Provided psychological momentum shift
- Dragon secured helped TSM's soul point race

**Could It Have Been Different?** 
100T overcommitted to a dragon they didn't need. With a 3.5k lead, trading dragon for a tower or enemy jungle camps would have been safer.

---

### Turning Point #2: Baron Throw (22:00) - Impact: CRITICAL

**What Happened:** 100T attempted Baron with TSM nearby. Baron was stolen and 100T was aced.

**Why It Mattered:**
- Complete gold lead evaporated
- TSM gained Baron buff despite being "behind"
- 100T's early game composition lost its window
- Mental devastation for 100T

**Could It Have Been Different?**
Absolutely. 100T had no business starting Baron without:
- Vision denial complete
- Knowledge of TSM jungler position
- Burst to finish Baron quickly

This was the game-losing mistake.

---

### Turning Point #3: Baron Secure (28:00) - Impact: HIGH

**What Happened:** TSM took Baron cleanly with numbers advantage from previous fight.

**Why It Mattered:**
- Formalized TSM's comeback
- With Baron, TSM's scaling became overwhelming
- Map control shifted completely
- Set up Elder Dragon control

---

### Turning Point Ranking

1. **Baron Throw (22:00)** - Single most impactful moment. Changed game state from "100T favored" to "TSM favored" instantly.

2. **Dragon Fight (18:00)** - Proved TSM could fight, set up the Baron scenario.

3. **Baron Secure (28:00)** - Sealed the deal but outcome was already trending TSM.

### Lesson
Early game leads require disciplined closing. One mechanical mistake at Baron negated 22 minutes of perfect play."""
            },
        ]
        
        training_samples.extend(turning_point_samples)
        
        # ============================================================
        # STRATEGIC RECOMMENDATIONS
        # ============================================================
        
        recommendation_samples = [
            {
                "instruction": "Provide detailed strategic recommendations for the losing team based on this match analysis.",
                "input": """Team: Team Liquid (Lost to Cloud9)
Match Result: 15-22, 35 minutes

What Happened:
- TL had early game comp (Renekton, Viego, Nautilus)
- TL executed early game well: first blood, first tower, 1.5k lead at 15
- TL lost crucial teamfight at dragon (18 min): Azir shuffle caught 3 members
- TL lost Baron fight (25 min): Thresh hook pulled Viego
- TL couldn't close out despite early advantages

Composition:
TL: Renekton, Viego, Viktor, Kai'Sa, Nautilus
C9: Gnar, Lee Sin, Azir, Jinx, Thresh""",
                "output": """## Strategic Recommendations for Team Liquid

### IMMEDIATE FIXES (Next Game)

**1. Dragon Fight Positioning**
The Azir shuffle that caught 3 members was preventable:
- Spread formation when fighting near walls
- Designate a player to watch for Azir angles
- Consider having Nautilus zone Azir rather than engaging

**2. Baron Vision Setup**
The Thresh hook came from fog:
- Clear surrounding vision BEFORE starting Baron
- Have support sweep while team postures
- Don't start Baron without knowing Thresh position

**3. Closing Execution**
With 1.5k lead at 15, the game should have been closed by 25 minutes:
- Force fights before Jinx completes 2 items
- Use Herald more aggressively for tower damage
- Don't give free scaling time

---

### SHORT-TERM IMPROVEMENTS (Next Week)

**Draft Considerations:**
1. If drafting early-game, ensure you have engage tools that work late
2. Nautilus is good but predictable - consider Rakan for similar engage with better scaling
3. Viktor as your "insurance" pick doesn't match well into Azir long-range

**Macro Strategy Adjustments:**
1. Practice 1-3-1 split with early game comps to avoid teamfighting
2. Drill Baron setups - vision, positioning, execution order
3. Create playbook for "ahead but can't teamfight" scenarios

**Team Coordination:**
1. Review teamfight comm recordings - were calls being made?
2. Assign dragon fight responsibilities clearly
3. Practice disengaging lost fights (you lost 4-1, should have been 1-0)

---

### LONG-TERM DEVELOPMENT (Next Month)

**Skills to Practice:**
- **Viego player:** Positioning when team is ahead - you got hooked because you were looking for plays when you should have been safe
- **Support player:** More aggressive ward clearing in mid-game to enable macro plays
- **Team:** Late game teamfighting even with early comps - sometimes games go long

**Patterns to Break:**
- Forcing 5v5 when composition doesn't support it
- Starting Baron without complete vision control
- Allowing free scaling without applying map pressure

**New Strategies to Develop:**
- Split push playbook for early-game compositions
- "Ahead but scaling" decision tree
- Dragon trading protocols (when to take vs. give)

---

### WARNING SIGNS TO WATCH

If in future games you see:
1. ✋ Enemy has Azir/Orianna with your team grouping tight
2. ✋ You're ahead but enemy comp is "waiting to scale"
3. ✋ Starting Baron without sweeping Thresh ward spots

**Stop and adjust.** These are the patterns that cost you this game."""
            },
        ]
        
        training_samples.extend(recommendation_samples)
        
        # ============================================================
        # DRAFT ANALYSIS
        # ============================================================
        
        draft_samples = [
            {
                "instruction": "Analyze the draft phase and team compositions, explaining win conditions for each team.",
                "input": """Game: League of Legends (Pro Play)

Ban Phase:
Blue Bans: Ahri, Maokai, Rek'Sai, Xayah, Rakan
Red Bans: Senna, Jax, K'Sante, Orianna, Ashe

Draft Order:
B1: Jinx
R1-R2: Renekton, Nautilus
B2-B3: Lee Sin, Thresh
R3: Viego
B4-B5: Azir, Gnar
R4-R5: Viktor, Kai'Sa

Final Compositions:
Blue: Gnar, Lee Sin, Azir, Jinx, Thresh
Red: Renekton, Viego, Viktor, Kai'Sa, Nautilus""",
                "output": """## Draft Analysis

### Blue Side Draft Strategy

**Pick 1 - Jinx (Power Pick)**
Jinx first pick signals a clear intent: we want to play late game. This forces Red to respond - do they try to punish the scaling ADC or draft their own late game?

**Picks 2-3 - Lee Sin, Thresh**
Both are flex picks that don't reveal much. Lee Sin can be early tempo or peel-focused. Thresh provides both catch potential and Jinx protection.

**Picks 4-5 - Azir, Gnar**
Doubles down on scaling. Azir is the ultimate late-game mid laner with team fight-winning shuffle potential. Gnar provides flexible top lane with Mega Gnar team fight control.

**Blue Side Win Condition:**
Scale to 25+ minutes, win one teamfight with Azir shuffle + Jinx cleanup. Protect Jinx at all costs.

---

### Red Side Draft Strategy

**Picks 1-2 - Renekton, Nautilus**
Immediate early game response to Jinx. Renekton wants to dominate before teams group. Nautilus provides point-and-click lockdown to catch Jinx in lane or skirmishes.

**Pick 3 - Viego**
Strong early-game jungler who can snowball with Renekton. Reset mechanic can win early fights convincingly.

**Picks 4-5 - Viktor, Kai'Sa**
Interesting choice - these picks ADD scaling rather than commit fully to early game. Viktor scales well, Kai'Sa is a late-game hypercarry.

**Red Side Win Condition:**
Dominate 0-20 minutes, build insurmountable lead through dives and early dragons. Close before Jinx reaches 3 items.

---

### Draft Grades

**Blue Side: A-**
- ✅ Clear identity and win condition
- ✅ Multiple teamfight tools (Azir R, Gnar R, Thresh R)
- ✅ Excellent scaling
- ⚠️ Vulnerable early game
- ⚠️ If Jinx dies in fights, damage disappears

**Red Side: B**
- ✅ Strong early game
- ✅ Good catch potential (Nautilus ult, Viego)
- ⚠️ Mixed identity - half early, half scaling
- ❌ Viktor doesn't fit the "end early" plan
- ❌ No answer to Azir shuffle late game

---

### Key Draft Observations

1. **Red Side's Viktor Pick is Questionable**
If you want to end early against scaling, pick something with more early pressure. Syndra, LeBlanc, or even Twisted Fate would create more windows.

2. **Blue Side's Lee Sin as Insurance**
Lee Sin provides early game relevance while Blue waits to scale. Smart pickup that doesn't sacrifice late-game team fighting.

3. **Kai'Sa vs Jinx Scaling**
In pure late game, Jinx out-scales Kai'Sa in teamfights (range advantage, AoE). Red needed to not let it get there.

### Predicted Outcome
Based purely on draft: **Blue Side favored** if game goes past 25 minutes. Red Side must execute their early window flawlessly."""
            },
        ]
        
        training_samples.extend(draft_samples)
        
        # ============================================================
        # NARRATIVE STYLE SUMMARIES
        # ============================================================
        
        narrative_samples = [
            {
                "instruction": "Write an engaging narrative-style recap of this match for esports fans.",
                "input": """Match: Grand Finals, World Championship
Teams: Dynasty vs Phoenix
Game: League of Legends
Duration: 48 minutes
Winner: Phoenix

Key Moments:
- Phoenix down 0-2 in series, facing elimination
- Dynasty leads 8k gold at 30 minutes
- Baron stolen by Phoenix jungler
- Elder Dragon fight: Phoenix wins 5-0 ace
- Phoenix completes reverse sweep""",
                "output": """# The Miracle Run: Phoenix Rises from the Ashes

In what will be remembered as one of the greatest comebacks in esports history, Phoenix completed the impossible reverse sweep against Dynasty in the World Championship Grand Finals.

## The Brink of Defeat

Down 0-2 in the series and staring elimination in the face, Phoenix found themselves in another desperate situation. Dynasty, the tournament favorites, had built an 8,000 gold lead by 30 minutes. The gold graph looked like a ski slope - one direction, no return.

Analysts had already started writing the coronation. Dynasty was one teamfight away from lifting the trophy. The crowd, predominantly Dynasty fans, began their victory chants.

Then everything changed.

## The Steal That Shook the World

Phoenix's jungler, known for his calculated plays, threw caution to the wind. As Dynasty started Baron, he flash-smited over the wall - a 50/50 that would either extend the game or end Phoenix's championship dreams.

**The smite connected.**

The arena erupted. Phoenix secured the Baron against all odds, and suddenly that 8,000 gold lead meant nothing compared to the purple buff glowing on five Phoenix players.

## The Elder Dragon Stand

But Baron alone wouldn't be enough. Phoenix needed the Elder Dragon to have any chance at cracking Dynasty's base. Both teams knew this. Everything came down to one fight at the dragon pit.

Dynasty engaged first. They had the gold lead, the better positioning, and momentum from two previous wins. By all metrics, they should have won.

They didn't win a single kill.

Phoenix executed the most perfect teamfight of the tournament - abilities sequenced flawlessly, focus fire coordinated to perfection, and when the dust settled, five Dynasty players lay defeated while all five Phoenix members stood.

**0-5. Complete devastation.**

## The Championship Moment

With Elder Dragon empowering their attacks, Phoenix marched down mid lane. Dynasty, respawning one by one, could only watch as their nexus crumbled.

Phoenix had done it. Not just won the game - they won the series 3-2, completing a reverse sweep that will be talked about for generations.

## The Legacy

Some say Phoenix had no business winning that game. The numbers said Dynasty should have closed. The history said no team comes back from 0-2 in Finals.

But esports isn't played on spreadsheets. It's played by competitors who refuse to accept defeat. And on that day, Phoenix showed the world what champions are made of.

*Never count them out.*"""
            },
        ]
        
        training_samples.extend(narrative_samples)
        
        # ============================================================
        # TECHNICAL ANALYSIS
        # ============================================================
        
        technical_samples = [
            {
                "instruction": "Provide a technical breakdown of the teamfight execution in this match.",
                "input": """Game: League of Legends
Fight: Dragon Pit, 24:00
Teams:
- Blue: Gnar, Lee Sin, Azir, Jinx, Thresh
- Red: Renekton, Viego, Viktor, Kai'Sa, Nautilus

Pre-fight State:
- Red starting dragon (Infernal)
- Blue approaching from river
- Dragon at 2,000 HP when fight starts

Fight Breakdown:
1. Thresh hooks Viego (pulls him out of pit)
2. Lee Sin kicks Renekton into Red backline
3. Azir shuffles 3 (Viktor, Kai'Sa, Nautilus) into Blue team
4. Jinx rockets clean up
5. Fight ends 5-1 for Blue""",
                "output": """## Technical Teamfight Breakdown

### Pre-Fight Analysis

**Positioning Assessment:**
- Red's formation for dragon was SUBOPTIMAL: Viego tanking dragon, Renekton zoning river, Viktor/Kai'Sa/Nautilus clustered behind pit wall
- Blue's approach was CORRECT: Thresh leading for hook angles, Azir maintaining shuffle range, Jinx safe at max auto range

**Cooldown States (estimated):**
- Thresh Hook: Available (critical)
- Lee Sin R: Available
- Azir R: Available (game-deciding)
- Nautilus R: Available (never used effectively)

---

### Fight Sequence Breakdown

**[0.0s] - Thresh Hook on Viego**
- EXCELLENT target selection. Viego in pit = isolated from team
- Hook angle from river brush denied Nautilus counter-engage
- Landed at max range - Thresh positioned aggressively but safely
- Viego's options: Ult for reset (risky), Flash (used), or Die

**[0.5s] - Viego Flash + Death**
- Viego flashed toward Red team but was followed by Lee Sin
- Lee Sin Q-Q follow guaranteed the kill
- This removed Red's only reset threat from the fight
- 5v4 before main engage

**[1.2s] - Lee Sin Insec on Renekton**
- With Viego dead, Lee Sin had freedom to reposition
- Ward-hop behind Renekton, R into Red backline
- Renekton knockback disrupted Red's formation
- More importantly: Renekton now BEHIND his carries

**[1.8s] - Azir Shuffle (Fight-Winning Play)**

This was the mechanical crux of the fight:
- Azir positioned near dragon pit wall
- W-E-Q-R combo executed in 0.4 seconds
- Targets hit: Viktor, Kai'Sa, Nautilus (3/4 remaining enemies)
- Displacement: All three pushed TOWARD Blue team, AWAY from safety
- Nautilus ult was buffered but animation cancelled by displacement

**Technical Note:** Azir's shuffle timing was frame-perfect. Any earlier and Viktor would have phase rushed away. Any later and Nautilus ult would have hit Jinx.

**[2.5s] - Cleanup Phase**
- 3 Red members at 60% HP, displaced into Blue team
- Jinx Fishbones rockets + Runaan's hit all three
- Thresh E (Flay) interrupted Kai'Sa E dash
- Viktor died first (no Zhonya's available)
- Kai'Sa died second (Thresh flay prevented escape)
- Nautilus died third (no priority target to ult)

**[4.0s] - Renekton Cleanup**
- Renekton tried to engage back in (Slice and Dice + Flash)
- Gnar Mega-transformed, R into wall
- Renekton died without dealing meaningful damage

---

### Technical Grades

| Player | Grade | Reasoning |
|--------|-------|-----------|
| Thresh | A+ | Perfect hook target, angle, and follow-up Flay |
| Lee Sin | A | Good Insec, though could have kicked Kai'Sa instead |
| Azir | S | Frame-perfect shuffle, won the fight solo |
| Jinx | A- | Good positioning, solid cleanup damage |
| Gnar | B+ | Mega Gnar came at right time, clean finish |

### Red Side Errors

1. **Viktor:** No Zhonya's purchase despite gold for it
2. **Nautilus:** Should have ulted Thresh when hook animation started
3. **Kai'Sa:** Positioning too close to Viktor pre-fight
4. **Viego:** Taking dragon damage instead of zoning with team
5. **Renekton:** Over-extended from carries, couldn't peel"""
            },
        ]
        
        training_samples.extend(technical_samples)
        
        # ============================================================
        # Save Dataset
        # ============================================================
        
        print(f"   ✓ Created {len(training_samples)} training samples")
        
        # Create dataset
        ds = Dataset.from_list(training_samples)
        ds.save_to_disk(self.data_dir / "esports_processed")
        print(f"   ✓ Saved to {self.data_dir / 'esports_processed'}")
        
        # Also save as JSON for review
        with open(self.data_dir / "esports_samples.json", "w") as f:
            json.dump(training_samples, f, indent=2)
        print(f"   ✓ JSON backup saved to {self.data_dir / 'esports_samples.json'}")
        
        return ds
    
    def create_combined_dataset(self, dialogsum_samples: int = 500) -> DatasetDict:
        """Create combined training dataset."""
        print("\n📦 Creating combined dataset...")
        
        datasets_to_combine = []
        
        # Load or create DialogSum
        dialogsum_path = self.data_dir / "dialogsum_processed"
        if dialogsum_path.exists():
            print("   Loading existing DialogSum data...")
            dialogsum_ds = Dataset.load_from_disk(dialogsum_path)
            datasets_to_combine.append(dialogsum_ds)
        else:
            dialogsum_ds = self.download_dialogsum(max_samples=dialogsum_samples)
            if dialogsum_ds:
                datasets_to_combine.append(dialogsum_ds)
        
        # Load or create ESports
        esports_path = self.data_dir / "esports_processed"
        if esports_path.exists():
            print("   Loading existing ESports data...")
            esports_ds = Dataset.load_from_disk(esports_path)
            datasets_to_combine.append(esports_ds)
        else:
            esports_ds = self.create_esports_dataset()
            datasets_to_combine.append(esports_ds)
        
        # Combine
        if len(datasets_to_combine) > 1:
            combined = concatenate_datasets(datasets_to_combine)
        else:
            combined = datasets_to_combine[0]
        
        # Shuffle
        combined = combined.shuffle(seed=42)
        
        # Split
        split = combined.train_test_split(test_size=0.1, seed=42)
        
        final_dataset = DatasetDict({
            'train': split['train'],
            'validation': split['test']
        })
        
        # Save
        final_dataset.save_to_disk(self.data_dir / "combined_training")
        
        print(f"\n✅ Combined dataset created:")
        print(f"   Training samples: {len(final_dataset['train'])}")
        print(f"   Validation samples: {len(final_dataset['validation'])}")
        print(f"   Saved to: {self.data_dir / 'combined_training'}")
        
        return final_dataset


def main():
    parser = argparse.ArgumentParser(description="Prepare training data for ESports Strategy AI")
    parser.add_argument("--only-esports", action="store_true", help="Only create esports data, skip DialogSum")
    parser.add_argument("--samples", type=int, default=500, help="Number of DialogSum samples to use")
    parser.add_argument("--data-dir", type=str, default="./data", help="Directory to save data")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("   ESports Strategy AI - Data Preparation")
    print("=" * 60)
    
    prep = ESportsDataPreparation(data_dir=args.data_dir)
    
    if args.only_esports:
        prep.create_esports_dataset()
    else:
        prep.create_combined_dataset(dialogsum_samples=args.samples)
    
    print("\n✅ Data preparation complete!")


if __name__ == "__main__":
    main()
PREPARE_DATA_EOF

echo -e "${GREEN}✓ prepare_data.py created${NC}"

# ============================================================
# Create Training Script
# ============================================================
echo -e "${YELLOW}🏋️ Creating training script...${NC}"

cat > "$PROJECT_DIR/training/train.py" << 'TRAIN_EOF'
"""
Model Training Script for ESports Strategy AI
Fine-tunes an open-source LLM using LoRA for efficient training.

Supported Models:
- google/gemma-2b-it (default, efficient)
- meta-llama/Llama-2-7b-chat-hf
- mistralai/Mistral-7B-Instruct-v0.2
- tiiuae/falcon-7b-instruct

Usage:
    python train.py                          # Train with defaults
    python train.py --model gemma-2b         # Specify model
    python train.py --epochs 5 --batch-size 4
    python train.py --resume ./checkpoints/checkpoint-500
"""

import os
import sys
import json
import argparse
import torch
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List

from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel
)


# Available models
MODELS = {
    "gemma-2b": "google/gemma-2b-it",
    "gemma-7b": "google/gemma-7b-it",
    "llama2-7b": "meta-llama/Llama-2-7b-chat-hf",
    "llama3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.2",
    "falcon-7b": "tiiuae/falcon-7b-instruct",
}

# LoRA target modules per model
LORA_TARGETS = {
    "gemma": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "falcon": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
}


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Model
    model_name: str = "google/gemma-2b-it"
    
    # Training
    num_epochs: int = 3
    batch_size: int = 2
    gradient_accumulation: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    max_seq_length: int = 2048
    
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    # Quantization
    use_4bit: bool = True
    
    # Paths
    data_dir: str = "./data/combined_training"
    output_dir: str = "./models/esports-lora"
    checkpoint_dir: str = "./checkpoints"
    
    # Resume
    resume_from: Optional[str] = None


class ESportsTrainer:
    """Handles model training with LoRA."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
    def setup(self):
        """Setup model, tokenizer, and LoRA."""
        print("\n" + "=" * 60)
        print("   Setting up training environment")
        print("=" * 60)
        
        # Check GPU
        if torch.cuda.is_available():
            print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("⚠️  No GPU detected! Training will be very slow.")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                sys.exit(0)
        
        # Load tokenizer
        print(f"\n📥 Loading tokenizer: {self.config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Quantization config
        print("📦 Setting up 4-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config.use_4bit,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Load model
        print(f"📥 Loading model: {self.config.model_name}")
        print("   (This may take a few minutes...)")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        
        # Prepare for training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Setup LoRA
        print("🔧 Configuring LoRA adapters...")
        target_modules = self._get_target_modules()
        
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        trainable, total = self.model.get_nb_trainable_parameters()
        print(f"✅ Model setup complete!")
        print(f"   Trainable parameters: {trainable:,} ({100 * trainable / total:.2f}%)")
        print(f"   Total parameters: {total:,}")
    
    def _get_target_modules(self) -> List[str]:
        """Get LoRA target modules based on model architecture."""
        model_lower = self.config.model_name.lower()
        
        for key, targets in LORA_TARGETS.items():
            if key in model_lower:
                return targets
        
        # Default
        return ["q_proj", "v_proj"]
    
    def load_data(self):
        """Load and prepare training data."""
        print(f"\n📊 Loading dataset from {self.config.data_dir}")
        
        try:
            dataset = load_from_disk(self.config.data_dir)
            print(f"   ✅ Loaded {len(dataset['train'])} training samples")
            print(f"   ✅ Loaded {len(dataset['validation'])} validation samples")
        except Exception as e:
            print(f"   ❌ Error loading dataset: {e}")
            print("   Run prepare_data.py first!")
            sys.exit(1)
        
        # Format prompts
        print("📝 Formatting prompts...")
        
        def format_prompt(example):
            if example.get("input", "").strip():
                text = f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}</s>"""
            else:
                text = f"""### Instruction:
{example['instruction']}

### Response:
{example['output']}</s>"""
            return {"text": text}
        
        train_formatted = dataset['train'].map(format_prompt, remove_columns=dataset['train'].column_names)
        val_formatted = dataset['validation'].map(format_prompt, remove_columns=dataset['validation'].column_names)
        
        # Tokenize
        print("🔤 Tokenizing...")
        
        def tokenize(examples):
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.config.max_seq_length,
                padding="max_length",
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
        
        self.train_dataset = train_formatted.map(tokenize, batched=True, remove_columns=["text"])
        self.val_dataset = val_formatted.map(tokenize, batched=True, remove_columns=["text"])
        
        print(f"   ✅ Training samples: {len(self.train_dataset)}")
        print(f"   ✅ Validation samples: {len(self.val_dataset)}")
    
    def train(self):
        """Run training."""
        print("\n" + "=" * 60)
        print("   Starting Training")
        print("=" * 60)
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{self.config.output_dir}_{timestamp}"
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            logging_steps=10,
            save_strategy="steps",
            save_steps=100,
            eval_strategy="steps",
            eval_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=True,
            report_to="none",  # Set to "wandb" if using Weights & Biases
            optim="paged_adamw_8bit",
            gradient_checkpointing=True,
            max_grad_norm=0.3,
            group_by_length=True,
            save_total_limit=3,
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
        )
        
        # Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Resume from checkpoint if specified
        resume_checkpoint = None
        if self.config.resume_from:
            resume_checkpoint = self.config.resume_from
            print(f"📂 Resuming from checkpoint: {resume_checkpoint}")
        
        # Train!
        print("\n🚀 Training started!")
        print(f"   Epochs: {self.config.num_epochs}")
        print(f"   Batch size: {self.config.batch_size}")
        print(f"   Gradient accumulation: {self.config.gradient_accumulation}")
        print(f"   Effective batch size: {self.config.batch_size * self.config.gradient_accumulation}")
        print(f"   Learning rate: {self.config.learning_rate}")
        print()
        
        train_result = self.trainer.train(resume_from_checkpoint=resume_checkpoint)
        
        # Print results
        print("\n" + "=" * 60)
        print("   Training Complete!")
        print("=" * 60)
        print(f"   Training loss: {train_result.metrics['train_loss']:.4f}")
        
        # Save final model
        final_path = f"{self.config.output_dir}_final"
        print(f"\n💾 Saving model to {final_path}")
        self.model.save_pretrained(final_path)
        self.tokenizer.save_pretrained(final_path)
        
        # Save config
        config_path = Path(final_path) / "training_config.json"
        with open(config_path, "w") as f:
            json.dump(vars(self.config), f, indent=2)
        
        print(f"✅ Model saved successfully!")
        print(f"\nTo use your trained model:")
        print(f"   1. Set LOCAL_MODEL_PATH={final_path} in .env")
        print(f"   2. Set LLM_BACKEND=local in .env")
        print(f"   3. Restart the server")
        
        return train_result
    
    def test_generation(self, prompt: str = None):
        """Test the trained model with a sample prompt."""
        print("\n🧪 Testing model generation...")
        
        if prompt is None:
            prompt = "Summarize a League of Legends match where the blue team won through superior teamfighting."
        
        formatted = f"""### Instruction:
{prompt}

### Response:
"""
        
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "### Response:" in response:
            response = response.split("### Response:")[-1].strip()
        
        print(f"\nPrompt: {prompt}")
        print(f"\nGenerated Response:\n{response}")


def main():
    parser = argparse.ArgumentParser(description="Train ESports Strategy AI model")
    parser.add_argument("--model", type=str, default="gemma-2b", 
                       choices=list(MODELS.keys()),
                       help="Model to fine-tune")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Training batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--data-dir", type=str, default="./data/combined_training", help="Training data path")
    parser.add_argument("--output-dir", type=str, default="./models/esports-lora", help="Output directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--test-only", action="store_true", help="Only test existing model")
    
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig(
        model_name=MODELS[args.model],
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        resume_from=args.resume,
    )
    
    print("=" * 60)
    print("   ESports Strategy AI - Model Training")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"   Model: {config.model_name}")
    print(f"   Epochs: {config.num_epochs}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   LoRA rank: {config.lora_r}")
    
    # Initialize trainer
    trainer = ESportsTrainer(config)
    
    # Setup
    trainer.setup()
    
    if not args.test_only:
        # Load data
        trainer.load_data()
        
        # Train
        trainer.train()
    
    # Test
    trainer.test_generation()
    
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
TRAIN_EOF

echo -e "${GREEN}✓ train.py created${NC}"

# ============================================================
# Create Evaluation Script
# ============================================================
echo -e "${YELLOW}📊 Creating evaluation script...${NC}"

cat > "$PROJECT_DIR/training/evaluate.py" << 'EVAL_EOF'
"""
Evaluation Script for ESports Strategy AI
Tests model quality using ROUGE scores and custom metrics.

Usage:
    python evaluate.py --model ./models/esports-lora_final
    python evaluate.py --model ./models/esports-lora_final --detailed
"""

import os
import json
import argparse
import torch
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_from_disk
import numpy as np


class ModelEvaluator:
    """Evaluates trained model quality."""
    
    def __init__(self, model_path: str, base_model: str = None):
        self.model_path = Path(model_path)
        self.model = None
        self.tokenizer = None
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Detect base model from config
        config_path = self.model_path / "training_config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
                base_model = config.get("model_name", base_model)
        
        self.base_model = base_model or "google/gemma-2b-it"
        
    def load_model(self):
        """Load the trained model."""
        print(f"📥 Loading model from {self.model_path}")
        
        # Quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Load base model
        print(f"   Loading base: {self.base_model}")
        base = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load LoRA
        print(f"   Loading LoRA adapters")
        self.model = PeftModel.from_pretrained(base, str(self.model_path))
        self.model.eval()
        
        print("✅ Model loaded!")
    
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate response from prompt."""
        formatted = f"""### Instruction:
{prompt}

### Response:
"""
        
        inputs = self.tokenizer(formatted, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "### Response:" in response:
            response = response.split("### Response:")[-1].strip()
        
        return response
    
    def evaluate_rouge(self, generated: str, reference: str) -> Dict[str, float]:
        """Calculate ROUGE scores."""
        scores = self.scorer.score(reference, generated)
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure,
        }
    
    def evaluate_dataset(self, data_path: str, max_samples: int = 50) -> Dict:
        """Evaluate on dataset."""
        print(f"\n📊 Evaluating on {data_path}")
        
        try:
            dataset = load_from_disk(data_path)
            eval_data = dataset['validation']
        except:
            print("   ❌ Could not load dataset")
            return {}
        
        results = []
        
        for i, sample in enumerate(tqdm(eval_data[:max_samples], desc="Evaluating")):
            # Generate
            prompt = sample['instruction']
            if sample.get('input'):
                prompt += f"\n\n{sample['input']}"
            
            generated = self.generate(prompt)
            reference = sample['output']
            
            # Score
            scores = self.evaluate_rouge(generated, reference)
            scores['sample_id'] = i
            results.append(scores)
        
        # Aggregate
        avg_scores = {
            'avg_rouge1': np.mean([r['rouge1'] for r in results]),
            'avg_rouge2': np.mean([r['rouge2'] for r in results]),
            'avg_rougeL': np.mean([r['rougeL'] for r in results]),
            'num_samples': len(results),
        }
        
        return avg_scores
    
    def run_test_cases(self) -> List[Dict]:
        """Run predefined test cases."""
        print("\n🧪 Running test cases...")
        
        test_cases = [
            {
                "name": "Match Summary",
                "prompt": "Summarize a League of Legends match where the blue team came back from a 5k gold deficit to win through superior teamfighting at Baron.",
            },
            {
                "name": "Turning Points",
                "prompt": "Identify the key turning points when a team loses despite having an early game lead.",
            },
            {
                "name": "Recommendations",
                "prompt": "What strategic recommendations would you give to a team that keeps losing late-game teamfights despite having good compositions?",
            },
            {
                "name": "Draft Analysis",
                "prompt": "Analyze a team composition of Gnar, Lee Sin, Azir, Jinx, Thresh. What are the win conditions?",
            },
        ]
        
        results = []
        
        for tc in test_cases:
            print(f"\n   Testing: {tc['name']}")
            response = self.generate(tc['prompt'])
            
            results.append({
                "name": tc['name'],
                "prompt": tc['prompt'],
                "response": response,
                "response_length": len(response),
            })
            
            print(f"   Response length: {len(response)} chars")
            print(f"   Preview: {response[:200]}...")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--data", type=str, default="./data/combined_training", help="Evaluation data path")
    parser.add_argument("--samples", type=int, default=50, help="Number of samples to evaluate")
    parser.add_argument("--detailed", action="store_true", help="Show detailed test results")
    parser.add_argument("--output", type=str, default="./evaluation_results.json", help="Output file")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("   ESports Strategy AI - Model Evaluation")
    print("=" * 60)
    
    # Initialize
    evaluator = ModelEvaluator(args.model)
    evaluator.load_model()
    
    # Run evaluation
    all_results = {}
    
    # ROUGE evaluation
    rouge_results = evaluator.evaluate_dataset(args.data, max_samples=args.samples)
    all_results['rouge_scores'] = rouge_results
    
    print("\n📊 ROUGE Scores:")
    print(f"   ROUGE-1: {rouge_results.get('avg_rouge1', 0):.4f}")
    print(f"   ROUGE-2: {rouge_results.get('avg_rouge2', 0):.4f}")
    print(f"   ROUGE-L: {rouge_results.get('avg_rougeL', 0):.4f}")
    
    # Test cases
    test_results = evaluator.run_test_cases()
    all_results['test_cases'] = test_results
    
    if args.detailed:
        print("\n" + "=" * 60)
        print("   Detailed Test Results")
        print("=" * 60)
        for tc in test_results:
            print(f"\n### {tc['name']}")
            print(f"Prompt: {tc['prompt']}")
            print(f"\nResponse:\n{tc['response']}")
            print("-" * 40)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n💾 Results saved to {args.output}")
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
EVAL_EOF

echo -e "${GREEN}✓ evaluate.py created${NC}"

# ============================================================
# Create Training Runner Script
# ============================================================
echo -e "${YELLOW}🏃 Creating runner script...${NC}"

cat > "$PROJECT_DIR/training/run_training.sh" << 'RUN_TRAIN_EOF'
#!/bin/bash

# ESports Strategy AI - Training Runner
# This script runs the complete training pipeline

echo "============================================================"
echo "    ESports Strategy AI - Training Pipeline"
echo "============================================================"
echo ""

# Activate virtual environment if exists
if [ -d "../venv" ]; then
    source ../venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

# Step 1: Prepare Data
echo "📊 Step 1: Preparing training data..."
python prepare_data.py --samples 500

if [ $? -ne 0 ]; then
    echo "❌ Data preparation failed!"
    exit 1
fi

echo ""
echo "✅ Data preparation complete!"
echo ""

# Step 2: Train Model
echo "🏋️ Step 2: Training model..."
echo "   This will take a while depending on your GPU..."
echo ""

python train.py \
    --model gemma-2b \
    --epochs 3 \
    --batch-size 2 \
    --lr 2e-4 \
    --lora-r 16

if [ $? -ne 0 ]; then
    echo "❌ Training failed!"
    exit 1
fi

echo ""
echo "✅ Training complete!"
echo ""

# Step 3: Evaluate
echo "📊 Step 3: Evaluating model..."

# Find the latest model
LATEST_MODEL=$(ls -td ./models/esports-lora_final 2>/dev/null | head -1)

if [ -z "$LATEST_MODEL" ]; then
    LATEST_MODEL=$(ls -td ./models/esports-lora_* 2>/dev/null | head -1)
fi

if [ -n "$LATEST_MODEL" ]; then
    python evaluate.py --model "$LATEST_MODEL" --samples 20
else
    echo "⚠️  No trained model found to evaluate"
fi

echo ""
echo "============================================================"
echo "    Training Pipeline Complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "1. Check evaluation results in evaluation_results.json"
echo "2. Update .env with LOCAL_MODEL_PATH=$LATEST_MODEL"
echo "3. Set LLM_BACKEND=local in .env"
echo "4. Restart the backend server"
echo ""
RUN_TRAIN_EOF

chmod +x "$PROJECT_DIR/training/run_training.sh"

echo -e "${GREEN}✓ run_training.sh created${NC}"

# ============================================================
# Create Install Script
# ============================================================
cat > "$PROJECT_DIR/training/install_training.sh" << 'INSTALL_TRAIN_EOF'
#!/bin/bash

echo "============================================================"
echo "    Installing Training Dependencies"
echo "============================================================"

# Check Python
python3 --version

# Create venv if needed
if [ ! -d "venv" ] && [ ! -d "../venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
fi

# Activate
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d "../venv" ]; then
    source ../venv/bin/activate
fi

# Install
echo "Installing packages..."
pip install --upgrade pip
pip install -r requirements_training.txt

echo ""
echo "✅ Installation complete!"
echo ""
echo "To start training, run:"
echo "   ./run_training.sh"
echo ""
INSTALL_TRAIN_EOF

chmod +x "$PROJECT_DIR/training/install_training.sh"

echo -e "${GREEN}✓ install_training.sh created${NC}"

# ============================================================
# Summary
# ============================================================
echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}    Training Setup Complete!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo "Files created in training/:"
echo "  📄 prepare_data.py       - Data preparation script"
echo "  📄 train.py              - Model training script"
echo "  📄 evaluate.py           - Evaluation script"
echo "  📄 requirements_training.txt"
echo "  📄 run_training.sh       - Full pipeline runner"
echo "  📄 install_training.sh   - Dependency installer"
echo ""
echo -e "${YELLOW}To train a model:${NC}"
echo ""
echo "1. Install dependencies:"
echo "   ${BLUE}cd training && ./install_training.sh${NC}"
echo ""
echo "2. Run the full training pipeline:"
echo "   ${BLUE}./run_training.sh${NC}"
echo ""
echo "   Or run steps individually:"
echo "   ${BLUE}python prepare_data.py${NC}"
echo "   ${BLUE}python train.py --model gemma-2b --epochs 3${NC}"
echo "   ${BLUE}python evaluate.py --model ./models/esports-lora_final${NC}"
echo ""
echo -e "${YELLOW}Requirements:${NC}"
echo "  - NVIDIA GPU with CUDA support (8GB+ VRAM recommended)"
echo "  - ~20GB disk space for model and data"
echo "  - Python 3.9+"
echo ""
echo -e "${GREEN}============================================================${NC}"