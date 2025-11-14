# backend/prompt_templates.py

def format_events(events):
    """Helper to format events for prompts - handles missing fields gracefully"""
    if not events:
        return "No events recorded"
    
    formatted = []
    for i, event in enumerate(events):
        try:
            time = event.get('time', f'Event {i+1}')
            event_type = event.get('type', 'unknown')
            team = event.get('team', 'N/A')
            details = event.get('details', '')
            
            event_str = f"- {time}: {event_type}"
            
            if team and team != 'N/A':
                event_str += f" ({team})"
            
            if details:
                event_str += f" - {details}"
            
            formatted.append(event_str)
            
        except Exception as e:
            print(f"Warning: Could not format event {i}: {e}")
            formatted.append(f"- Event {i+1}: [formatting error]")
            continue
    
    return '\n'.join(formatted) if formatted else "No events could be formatted"


class PromptTemplates:
    
    @staticmethod
    def _get_length_instruction(length):
        """Get length instructions for prompts"""
        instructions = {
            "short": "Keep the analysis concise - aim for 500-800 words total.",
            "medium": "Provide a comprehensive analysis - aim for 1000-1500 words total.",
            "long": "Provide a detailed analysis - aim for 1500-2500 words total.",
            "detailed": "Provide an in-depth, thorough analysis - aim for 2500-4000 words total with extensive detail."
        }
        return instructions.get(length, instructions["medium"])
    
    @staticmethod
    def _get_focus_instruction(focus_areas):
        """Get focus area instructions"""
        if not focus_areas or "overall" in focus_areas:
            return "Cover all aspects comprehensively including team fights, economy, objectives, positioning, and draft."
        
        focus_map = {
            "team_fights": "Pay special attention to team fight analysis, positioning, engagement timing, and fight outcomes.",
            "economy": "Pay special attention to economic advantages, gold leads, farming efficiency, and resource control.",
            "objectives": "Pay special attention to objective control (dragons, barons, towers, inhibitors) and timing.",
            "positioning": "Pay special attention to map positioning, vision control, rotations, and map pressure.",
            "draft": "Pay special attention to draft analysis, team compositions, win conditions, and champion synergies."
        }
        
        instructions = []
        for focus in focus_areas:
            if focus in focus_map:
                instructions.append(focus_map[focus])
        
        return " ".join(instructions) if instructions else "Cover all aspects comprehensively."
    
    @staticmethod
    def _get_style_instruction(style):
        """Get style instructions"""
        styles = {
            "analytical": "Use an analytical, data-driven tone with clear structure and factual observations.",
            "narrative": "Use a narrative, story-telling tone that makes the match exciting and engaging while maintaining analytical depth.",
            "technical": "Use technical esports terminology and deep strategic analysis for experienced players.",
            "casual": "Use an accessible, casual tone that's easy to understand for newcomers while still being insightful."
        }
        return styles.get(style, styles["analytical"])
    
    @staticmethod
    def comprehensive_match_analysis(match_data, length="medium", focus=None, style="analytical", focus_team="team_a"):
        """
        Generate a comprehensive match analysis that includes:
        1. Match Overview & Summary
        2. Critical Turning Points Analysis
        3. Tactical Recommendations for focus team
        4. Key Takeaways
        """
        focus = focus or ["overall"]
        
        length_instruction = PromptTemplates._get_length_instruction(length)
        focus_instruction = PromptTemplates._get_focus_instruction(focus)
        style_instruction = PromptTemplates._get_style_instruction(style)
        
        # Safely get data with defaults
        match_id = match_data.get('match_id', 'Unknown')
        teams = match_data.get('teams', {})
        team_a = teams.get('team_a', 'Team A')
        team_b = teams.get('team_b', 'Team B')
        
        hero_picks = match_data.get('hero_picks', {})
        picks_a = hero_picks.get('team_a', [])
        picks_b = hero_picks.get('team_b', [])
        
        events = match_data.get('events', [])
        commentary = match_data.get('commentary', 'No commentary available.')
        
        focus_team_name = teams.get(focus_team, focus_team)
        other_team = 'team_b' if focus_team == 'team_a' else 'team_a'
        other_team_name = teams.get(other_team, 'opponent')
        
        picks_focus = hero_picks.get(focus_team, [])
        picks_other = hero_picks.get(other_team, [])
        
        prompt = f"""You are an expert esports analyst providing a comprehensive match analysis report.

{style_instruction}
{length_instruction}
{focus_instruction}

## Match Information

Match ID: {match_id}
Teams: {team_a} vs {team_b}

Hero/Character Picks:
- {team_a}: {', '.join(picks_a) if picks_a else 'Not specified'}
- {team_b}: {', '.join(picks_b) if picks_b else 'Not specified'}

Key Events Timeline:
{format_events(events)}

Match Commentary:
{commentary[:1000]}

---

## Your Analysis Should Include:

### 1. MATCH OVERVIEW & SUMMARY (2-3 paragraphs)
Provide a high-level summary of how the match unfolded. Cover:
- Opening phase and early game developments
- Mid-game momentum and key battles
- Late game/end game progression
- Final outcome and overall match flow

### 2. CRITICAL TURNING POINTS (Identify 3-5 key moments)
For each turning point, analyze:
- **Timestamp & Event**: When and what happened
- **Context**: Game state before this moment
- **Why It Mattered**: How this changed the match trajectory  
- **Impact**: Immediate and long-term effects on the game
- **Alternative Outcomes**: What could have happened differently

### 3. TACTICAL RECOMMENDATIONS FOR {focus_team_name} (5-7 recommendations)
Provide specific, actionable recommendations:

For each recommendation include:
- **Issue/Weakness Identified**: What went wrong or could improve
- **Tactical Recommendation**: Specific actionable advice
- **Expected Impact**: How this would improve performance
- **Implementation**: How to practice or execute this improvement

Focus on:
- Draft/composition improvements
- Early game strategy adjustments  
- Mid-game decision making
- Objective timing and control
- Team fighting and positioning
- Communication and coordination

### 4. KEY TAKEAWAYS (2-3 bullet points)
Summarize the most important lessons from this match for both teams and viewers.

---

**Important Instructions:**
- Be specific with timestamps and player/team actions
- Support claims with evidence from the events and commentary
- Make recommendations realistic and actionable
- {length_instruction}
- Keep analysis structured with clear headers
- Focus primarily on {focus_team_name}'s perspective for recommendations
"""
        return prompt