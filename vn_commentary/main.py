"""
Main driver program for visual novel commentary system.

Usage:
    python -m vn_commentary.main <dialogue_file.json>
    python -m vn_commentary.main <dialogue_file.json> --config vn_commentary/config.yaml

Output Policy:
    - print(): User-facing output (reactions, summaries, chapter headers)
    - logger.info(): Important events (file loading, saving results)
    - logger.debug(): Verbose diagnostics (processing progress, internal state)
"""
import asyncio
import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional
import yaml
import sys

# Ensure UTF-8 encoding for stdout/stderr on Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from core.logging import get_logger, setup_logging
from vn_commentary.dialogue_reader import DialogueReader
from vn_commentary.commentary_analyzer import CommentaryAnalyzer
from vn_commentary.models import CommentaryResult
from core.settings import LLMSettings

logger = get_logger(__name__)


class Colors:
    """ANSI color codes for terminal output."""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

    # Dialogue colors
    SPEAKER = '\033[36m'  # Cyan
    TEXT = '\033[97m'     # Bright white
    NARRATOR = '\033[35m' # Magenta

    # Reaction colors
    REACTION_PREFIX = '\033[33m'  # Yellow
    REACTION_TEXT = '\033[93m'    # Bright yellow

    # UI elements
    HEADER = '\033[94m'   # Blue
    SEPARATOR = '\033[90m'  # Gray

    @staticmethod
    def strip_if_no_color():
        """Check if colors should be disabled (for file redirection)."""
        import sys
        if not sys.stdout.isatty():
            # Disable colors when output is redirected
            for attr in dir(Colors):
                if not attr.startswith('_') and attr != 'strip_if_no_color':
                    setattr(Colors, attr, '')


class VNCommentaryDriver:
    """Main driver for visual novel commentary system."""

    def __init__(self, config_path: str = "vn_commentary/config.yaml"):
        """
        Initialize driver with configuration.

        Args:
            config_path: Path to configuration YAML file
        """
        self.config = self._load_config(config_path)
        self._setup_logging()
        self.analyzer: Optional[CommentaryAnalyzer] = None
        self.results: List[CommentaryResult] = []

        # Initialize colors (disable if output is redirected)
        Colors.strip_if_no_color()

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        config_file = Path(config_path)
        if not config_file.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return self._default_config()

        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        return config

    def _default_config(self) -> dict:
        """Return default configuration."""
        return {
            "llm": {
                "provider": "gemini",
                "providers": {
                    "gemini": {
                        "model": "gemini-2.5-flash",
                        "api_key": None,
                        "disable_thinking": False,
                        "generation_config": {
                            "temperature": 0.7
                        }
                    }
                }
            },
            "system_prompt_file": None,
            "output": {
                "log_level": "INFO",
                "save_results": True,
                "results_file": "commentary_results.json"
            },
            "processing": {
                "delay_between_dialogues": 0.0
            }
        }

    def _setup_logging(self):
        """Setup logging based on configuration."""
        log_level = self.config.get("output", {}).get("log_level", "INFO")
        setup_logging(level=log_level)

        # Suppress noisy LLM provider logging
        logging.getLogger("google_genai.models").setLevel(logging.WARNING)

    def _create_analyzer(self) -> CommentaryAnalyzer:
        """Create and configure commentary analyzer."""
        llm_settings = LLMSettings.from_dict(self.config.get("llm", {}))

        # Load custom system prompt if specified
        system_prompt = None
        prompt_file = self.config.get("system_prompt_file")
        if prompt_file and Path(prompt_file).exists():
            with open(prompt_file, 'r', encoding='utf-8') as f:
                system_prompt = f.read()
            logger.info(f"Loaded custom system prompt from {prompt_file}")

        # Create analyzer
        analyzer = CommentaryAnalyzer(
            llm_settings=llm_settings,
            system_prompt=system_prompt
        )

        return analyzer

    async def process_dialogues(self, dialogue_file: str):
        """
        Process all dialogues from a file as a single chapter.

        Args:
            dialogue_file: Path to JSON file with dialogues
        """
        # Load dialogues
        reader = DialogueReader(dialogue_file)
        logger.info(f"Loaded {len(reader)} dialogues from {dialogue_file}")

        # Create analyzer if not already created (reuse across chapters)
        if self.analyzer is None:
            self.analyzer = self._create_analyzer()

        # Set chapter context (LLM will see lines progressively as we process)
        self.analyzer.set_chapter(reader.dialogues)
        logger.debug("Chapter loaded into analyzer")

        # Process each dialogue
        delay = self.config.get("processing", {}).get("delay_between_dialogues", 0.0)

        for i, dialogue in enumerate(reader, 1):
            logger.debug(f"Processing dialogue {i}/{len(reader)}: {dialogue.dialogue_id}")

            # Analyze dialogue
            result = await self.analyzer.analyze_dialogue(dialogue)
            self.results.append(result)

            # Print dialogue line (user-facing output, not logging)
            self._print_dialogue(dialogue, result)

            # Optional delay for rate limiting
            if delay > 0 and i < len(reader):
                await asyncio.sleep(delay)

        # Signal end of chapter
        self.analyzer.end_chapter()

        # Save results if configured (append mode for multi-chapter)
        if self.config.get("output", {}).get("save_results", True):
            self._save_results()

    def _print_dialogue(self, dialogue, result: CommentaryResult):
        """
        Print dialogue line with optional reaction in a nicely formatted, colored way.

        Args:
            dialogue: The dialogue to print
            result: Analysis result containing optional reaction
        """
        from vn_commentary.models import Dialogue

        # Determine if this is a narrator line
        is_narrator = dialogue.speaker in ["[Narrative]", "[旁白]"]

        # Print speaker and dialogue
        if is_narrator:
            # Narrator lines in magenta
            print(f"{Colors.NARRATOR}{dialogue.speaker}{Colors.RESET}")
            print(f"  {Colors.DIM}{dialogue.chinese_text}{Colors.RESET}")
        else:
            # Character dialogue
            print(f"{Colors.SPEAKER}{dialogue.speaker}:{Colors.RESET} {Colors.TEXT}{dialogue.chinese_text}{Colors.RESET}")

        # Print reaction if present
        if result.decision.action == "react" and result.decision.instruction:
            reaction_lines = result.decision.instruction.strip().split('\n')
            for line in reaction_lines:
                print(f"  {Colors.REACTION_PREFIX}💭{Colors.RESET} {Colors.REACTION_TEXT}{line}{Colors.RESET}")
            print(f"  {Colors.REACTION_PREFIX}💭{Colors.RESET} {Colors.REACTION_TEXT}{result.decision.mode} | {result.decision.emotion} | {result.decision.intensity}{Colors.RESET}")

        # Add spacing
        print()

    def _save_results(self):
        """Save analysis results to JSON file."""
        output_file = self.config.get("output", {}).get("results_file", "commentary_results.json")

        # Convert results to serializable format
        results_data = []
        for result in self.results:
            results_data.append({
                "dialogue_id": result.dialogue.dialogue_id,
                "speaker": result.dialogue.speaker,
                "chinese_text": result.dialogue.chinese_text,
                "japanese_text": result.dialogue.japanese_text,
                "action": result.decision.action,
                "reaction": result.decision.instruction,
                "reasoning": result.decision.reasoning
            })

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(results_data)} results to {output_file}")

    def _print_summary(self):
        """Print processing summary."""
        total = len(self.results)
        reactions = sum(1 for r in self.results if r.decision.action == "react")
        silent = total - reactions

        print(f"\n{Colors.SEPARATOR}{'=' * 60}{Colors.RESET}")
        print(f"{Colors.HEADER}{Colors.BOLD}COMMENTARY ANALYSIS SUMMARY{Colors.RESET}")
        print(f"{Colors.SEPARATOR}{'=' * 60}{Colors.RESET}")
        print(f"{Colors.TEXT}Total dialogues: {Colors.BOLD}{total}{Colors.RESET}")
        print(f"{Colors.REACTION_TEXT}Reactions: {Colors.BOLD}{reactions}{Colors.RESET} {Colors.DIM}({reactions/total*100:.1f}%){Colors.RESET}")
        print(f"{Colors.DIM}Silent: {silent} ({silent/total*100:.1f}%){Colors.RESET}")
        print(f"{Colors.SEPARATOR}{'=' * 60}{Colors.RESET}\n")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Visual Novel Commentary System - Analyze VN dialogues with LLM"
    )
    parser.add_argument(
        "dialogue_files",
        nargs="+",
        help="Path(s) to JSON file(s) containing visual novel dialogues (one file per chapter)"
    )
    parser.add_argument(
        "--config",
        default="vn_commentary/config.yaml",
        help="Path to configuration YAML file (default: vn_commentary/config.yaml)"
    )

    args = parser.parse_args()

    # Create driver
    driver = VNCommentaryDriver(config_path=args.config)

    # Process each chapter file
    for chapter_num, dialogue_file in enumerate(args.dialogue_files, 1):
        # Print chapter header (user-facing output)
        print(f"\n{Colors.SEPARATOR}{'='*60}{Colors.RESET}")
        print(f"{Colors.HEADER}{Colors.BOLD}PROCESSING CHAPTER {chapter_num}:{Colors.RESET} {Colors.TEXT}{dialogue_file}{Colors.RESET}")
        print(f"{Colors.SEPARATOR}{'='*60}{Colors.RESET}\n")
        await driver.process_dialogues(dialogue_file)

    # Final summary across all chapters
    if len(args.dialogue_files) > 1:
        driver._print_summary()


if __name__ == "__main__":
    asyncio.run(main())
