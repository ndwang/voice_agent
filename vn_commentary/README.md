# Visual Novel Commentary System

A standalone program that analyzes visual novel dialogues and generates contextual commentary using LLM-based analysis. The system reads dialogue chunks from JSON files and uses an LLM to determine if each line is worth commenting on.

## Features

- **LLM-Powered Analysis**: Uses Gemini (or other LLM providers) with structured output to make intelligent commentary decisions
- **Full Chapter Context**: LLM receives all dialogues in the chapter for better understanding of plot and character development
- **Smart Pacing**: Tracks reaction history and provides pacing feedback to avoid over-commenting or staying silent too long
- **Chapter-Aware Processing**: Supports chapter boundaries with proper state management
- **Configurable**: YAML-based configuration for LLM settings and output options
- **Reusable Components**: Leverages LLM providers and utilities from the main voice agent project
- **Structured Output**: Returns JSON-formatted decisions with action (silent/react), reaction text, and reasoning

## Architecture

The system consists of several components:

- **DialogueReader**: Loads and parses visual novel dialogues from JSON files
- **CommentaryAnalyzer**: Uses LLM to analyze dialogues with full chapter context and pacing awareness
  - Tracks current position in chapter
  - Monitors lines since last reaction
  - Provides pacing guidance to LLM
- **VNCommentaryDriver**: Main orchestrator that coordinates the pipeline and manages chapter boundaries

## Installation

The program reuses dependencies from the main voice agent project. Ensure you have:

1. Python 3.11+
2. `uv` for dependency management
3. Gemini API key (set as `GEMINI_API_KEY` environment variable or in config)

## Configuration

Edit `vn_commentary/config.yaml` to configure:

```yaml
llm:
  provider: gemini
  model: gemini-2.5-flash
  api_key: null  # Or set GEMINI_API_KEY environment variable
  temperature: 0.7

output:
  log_level: INFO
  save_results: true
  results_file: commentary_results.json

processing:
  delay_between_dialogues: 0.0  # For rate limiting
```

## Input Format

Dialogues should be in JSON format with the following structure:

```json
[
  {
    "dialogue_id": "0101Adv01_Narrative022",
    "speaker": "[Narrative]",
    "japanese_text": "唐突に答えを見つけた。",
    "chinese_text": "但某天她忽然发现了答案。"
  }
]
```

Or wrapped in an object:

```json
{
  "dialogues": [
    { ... }
  ]
}
```

## Usage

### Single Chapter

```bash
# From the project root directory
uv run python -m vn_commentary.main vn_commentary/example_dialogue.json
```

### Multiple Chapters

```bash
# Process multiple chapter files in sequence
uv run python -m vn_commentary.main chapter01.json chapter02.json chapter03.json
```

### With Custom Config

```bash
uv run python -m vn_commentary.main dialogues.json --config path/to/config.yaml
```

### Setting API Key

Option 1 - Environment variable:
```bash
export GEMINI_API_KEY="your-api-key-here"
uv run python -m vn_commentary.main dialogues.json
```

Option 2 - Config file:
```yaml
llm:
  api_key: "your-api-key-here"
```

## How It Works

### Progressive Chapter Context (No Spoilers!)

The analyzer loads the chapter file but only shows the LLM dialogues **up to the current position**. This means:
- The LLM experiences the story in real-time, just like a reader
- No spoilers - the LLM doesn't know what happens next
- Can reference all previous context from the chapter
- Reactions are natural and unspoiled

### Multi-Chapter Support

Process multiple chapters by providing multiple files:
```bash
uv run python -m vn_commentary.main chapter1.json chapter2.json chapter3.json
```

The analyzer maintains state across chapters, allowing continuous commentary throughout the story.

### Smart Pacing System

The system tracks reaction history and provides pacing feedback to the LLM:

- **Lines since last reaction**: Tells LLM how long it's been since the last comment
- **Pacing hints**:
  - Just reacted (0 lines): "avoid consecutive reactions unless crucial"
  - Long silence (8+ lines): "consider reacting to maintain engagement"
- **Context position**: Shows current position in chapter (e.g., "Line 7 of 12")

This prevents over-commenting on every line while ensuring the system doesn't stay silent for too long.

### Example Reasoning

With progressive context (no spoilers) and pacing awareness, the LLM provides natural, unspoiled reactions:

**Line 7 - Doesn't know what's coming:**
```json
{
  "reasoning": "This is a direct, natural follow-up question to the previous line about a 'special day.'
  Reacting to consecutive lines for such a minor conversational beat would be over-commenting.
  I'll wait for the answer to the protagonist's question, which will likely be more significant."
}
```

The LLM is genuinely curious about what the "special day" is - it doesn't know it's Emma's birthday yet!

## Output

The program outputs:

1. **Console Output**: Real-time reactions as they occur
   ```
   [0101Adv01_Ema008] 艾玛
     Line: 诶...哥哥,你该不会忘了吧?今天是...今天是我的生日啊!
     REACTION: Oh no! He really forgot her birthday! That's a huge oversight, poor Emma.
   ```

2. **Results File**: JSON file with all analysis results (if `save_results: true`)
   ```json
   [
     {
       "dialogue_id": "0101Adv01_Ema008",
       "speaker": "艾玛",
       "chinese_text": "诶...哥哥,你该不会忘了吧?今天是...今天是我的生日啊!",
       "japanese_text": "えっ...お兄ちゃん、まさか忘れたの?今日は...今日は私の誕生日だよ!",
       "action": "react",
       "reaction": "Oh no! He really forgot her birthday! That's a huge oversight, poor Emma.",
       "reasoning": "This is a major plot point and an emotional beat. The protagonist has forgotten Emma's birthday..."
     }
   ]
   ```

3. **Summary Statistics**: Processing summary at the end
   ```
   ============================================================
   COMMENTARY ANALYSIS SUMMARY
   ============================================================
   Total dialogues: 12
   Reactions: 4 (33.3%)
   Silent: 8 (66.7%)
   ============================================================
   ```

## Customization

### Custom System Prompt

Create a text file with your custom prompt and reference it in config:

```yaml
system_prompt_file: path/to/custom_prompt.txt
```

### Rate Limiting

Add delay between API calls to avoid rate limits:

```yaml
processing:
  delay_between_dialogues: 1.0  # 1 second delay
```

## Code Structure

```
vn_commentary/
├── __init__.py              # Package initialization
├── models.py                # Pydantic models (Dialogue, CommentaryDecision, etc.)
├── dialogue_reader.py       # JSON dialogue loader
├── commentary_analyzer.py   # LLM-based analysis engine
├── main.py                  # Main driver program
├── config.yaml             # Configuration file
├── example_dialogue.json   # Example input data
└── README.md               # This file
```

## Example

See `example_dialogue.json` for a sample visual novel chapter with 12 dialogue lines demonstrating various scenarios (mundane conversation, plot revelations, emotional moments).

## Extending

### Adding New LLM Providers

The system uses the `LLMProvider` abstraction from the main project. To add new providers:

1. Implement a provider class inheriting from `llm.base.LLMProvider`
2. Update `CommentaryAnalyzer._create_analyzer()` to support the new provider

### Customizing Decision Logic

Modify the system prompt in `CommentaryAnalyzer.DEFAULT_SYSTEM_PROMPT` or provide a custom prompt file to adjust when the system decides to react.

## Troubleshooting

### "Failed to parse JSON response"

The LLM sometimes returns non-JSON responses. The system includes fallback parsing that looks for keywords like "silent" or "react". Adjust the temperature or system prompt for more consistent JSON output.

### API Rate Limits

Use the `delay_between_dialogues` setting to add delays between requests:

```yaml
processing:
  delay_between_dialogues: 2.0  # 2 second delay
```

### Low Reaction Rate

If the system stays silent too often:
- Adjust the system prompt to be more reactive
- Lower the temperature for more deterministic decisions

## License

This component is part of the voice agent project and follows the same license.
