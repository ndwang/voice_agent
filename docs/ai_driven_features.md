# AI-Driven Features for Human-like Streaming Agent

## Core Philosophy
Let the LLM decide behaviors naturally instead of using deterministic triggers.
- ❌ Traditional: "If silence > 30s, trigger idle comment"
- ✅ AI-Driven: "LLM, it's been 45s since anyone spoke. Do you want to say something?"

---

## Feature 1: Spontaneous Speech System

### The Idea
Enable the agent to speak spontaneously without user input - proactive commentary, questions, and observations.

### Mechanism
**Background polling loop** that regularly asks the LLM if it wants to speak:

1. Every N seconds (configurable, e.g., 8s), gather rich context:
   - Time since last interaction
   - Stream stats (viewer count, chat rate)
   - Current emotional state from previous response
   - Recent conversation topics
   - Screen content (OCR)

2. Send lightweight prompt to LLM:
   ```
   "It's been {seconds} since anyone spoke. You can say something if you want, or stay quiet.
   Context: {stream_stats, time, emotion, recent_topics}
   What do you do? Reply: <speak>content</speak> OR <silent/> OR <wait/>"
   ```

3. LLM decides:
   - `<speak>` → Queue the message and process it normally
   - `<silent/>` → Continue waiting
   - `<wait/>` → Skip next polling cycle

### Why It Works
- LLM naturally learns conversation rhythm (more silence → lonely → more likely to speak)
- No hardcoded thresholds - behavior emerges from context interpretation
- Creates truly spontaneous, context-appropriate commentary
- Can use fast/cheap model for polling to reduce costs

### Examples of Emergent Behaviors
- Long silence → "静かだね...何か話そうか?"
- Just finished response → Asks follow-up question naturally
- Sees interesting screen content → Comments on it
- Remembers past topic → "あ、そういえば..."

---

## Feature 2: AI-Reported Emotional State

### The Idea
The agent feels and reports its own emotions - we don't calculate them with rules.

### Mechanism
**LLM self-reports emotional state** in its responses using tags:

1. Extend response format to include emotion/energy tags:
   ```
   <emotion>happy</emotion><energy>85</energy>
   <jp>わあ！急に人が増えた！</jp><zh>哇！突然来了好多人！</zh>
   ```

2. System extracts these tags and:
   - Stores current emotional state
   - Feeds it back in next prompt as context
   - Optionally displays in OBS overlay

3. Next prompt includes:
   ```
   "Your last reported state: Emotion: happy, Energy: 85%"
   ```

4. LLM instruction:
   ```
   "You can report how you're feeling emotionally using <emotion> and <energy> tags.
   This helps the system understand your state."
   ```

### Why It Works
- No emotion calculation engine ("happiness += 10" nonsense)
- LLM interprets context holistically and feels emotions naturally:
  - Lots of chat → excited/happy
  - Long silence → lonely/bored
  - Difficult questions → nervous/thoughtful
  - Long stream → tired/fatigued
- Emotions persist across turns for continuity
- More authentic than rule-based systems

### Key Difference
❌ Traditional: `emotion = base + (viewer_count * 0.1) - (interruptions * 5)`
✅ AI-Driven: LLM just IS emotional based on understanding the situation

---

## Feature 3: Episodic Memory System

### The Idea
Remember experiences and conversations, not just metadata - semantic, story-like memories.

### Mechanism
**LLM extracts and retrieves memories**, not keyword matching:

**Memory Creation:**
1. After conversations, ask LLM: "What's worth remembering from this interaction?"
2. LLM returns: `<memory>User mentioned learning programming</memory>`
3. Store to database with:
   - LLM-generated summary (natural language)
   - Timestamp
   - Emotional context at the time
   - Vector embedding (for semantic search)

**Memory Retrieval:**
1. When new conversation starts, compute embedding of current context
2. Vector search retrieves top N semantically similar memories
3. Inject into system prompt:
   ```
   "Previous memories:
   - [Someone asked about magic school last week]
   - [Talked about favorite games yesterday]
   - [Viewer shared they're learning Python]"
   ```
4. LLM references naturally when relevant: "前に誰かが聞いてたけど..."

### Why It Works
- Memories are semantic, not keyword-based (better matching)
- LLM decides when to reference memories (natural, not forced)
- Creates continuity across streams
- Builds depth over time - viewers return and are "remembered"

### What Gets Stored
- Conversation topics and themes
- Interesting facts or stories shared by viewers
- Milestones (first viewer, 100 viewer celebration)
- Emotional moments (funny jokes, touching messages)
- NOT just: "viewer123 visited 5 times"

### Examples
- Viewer returns: "あ、前にプログラミングの話してたよね？"
- Related topic: "これ、先週誰かが言ってたゲームと似てる！"
- Story callback: "魔法学校の話、前にしたっけ？"
