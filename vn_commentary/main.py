"""
Main driver program for visual novel commentary system.

Usage:
    python -m vn_commentary.main <dialogue_file.json>
    python -m vn_commentary.main <dialogue_file.json> --config vn_commentary/config.yaml
"""
import asyncio
import argparse
import json
from pathlib import Path
from typing import List, Optional
import yaml
import os
import sys

# Ensure UTF-8 encoding for stdout/stderr on Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from core.logging import get_logger, setup_logging
from vn_commentary.dialogue_reader import DialogueReader
from vn_commentary.commentary_analyzer import CommentaryAnalyzer
from vn_commentary.context_manager import ContextManager
from vn_commentary.models import CommentaryResult
from llm.providers.gemini import GeminiProvider

logger = get_logger(__name__)


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
                "model": "gemini-2.5-flash",
                "api_key": None,
                "temperature": 0.7
            },
            "context": {
                "max_context_size": 20
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

    def _create_analyzer(self) -> CommentaryAnalyzer:
        """Create and configure commentary analyzer."""
        llm_config = self.config.get("llm", {})
        context_config = self.config.get("context", {})

        # Load custom system prompt if specified
        system_prompt = None
        prompt_file = self.config.get("system_prompt_file")
        if prompt_file and Path(prompt_file).exists():
            with open(prompt_file, 'r', encoding='utf-8') as f:
                system_prompt = f.read()
            logger.info(f"Loaded custom system prompt from {prompt_file}")

        # Create LLM provider
        provider = llm_config.get("provider", "gemini")
        if provider == "gemini":
            llm_provider = GeminiProvider(
                model=llm_config.get("model", "gemini-2.5-flash"),
                api_key=llm_config.get("api_key") or os.getenv("GEMINI_API_KEY")
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        # Create context manager
        context_manager = ContextManager(
            max_context_size=context_config.get("max_context_size", 20)
        )

        # Create analyzer
        analyzer = CommentaryAnalyzer(
            llm_provider=llm_provider,
            context_manager=context_manager,
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

        # Create analyzer
        self.analyzer = self._create_analyzer()

        # Set full chapter context for better LLM understanding
        self.analyzer.set_chapter(reader.dialogues)
        logger.info("Chapter context loaded into analyzer")

        # Process each dialogue
        delay = self.config.get("processing", {}).get("delay_between_dialogues", 0.0)

        for i, dialogue in enumerate(reader, 1):
            logger.info(f"Processing dialogue {i}/{len(reader)}: {dialogue.dialogue_id}")

            # Analyze dialogue
            result = await self.analyzer.analyze_dialogue(dialogue)
            self.results.append(result)

            # Print result
            if result.decision.action == "react":
                print(f"\n[{dialogue.dialogue_id}] {dialogue.speaker}")
                print(f"  Line: {dialogue.chinese_text}")
                print(f"  REACTION: {result.decision.reaction}")
                print()

            # Optional delay for rate limiting
            if delay > 0 and i < len(reader):
                await asyncio.sleep(delay)

        # Signal end of chapter
        self.analyzer.end_chapter()

        # Save results if configured
        if self.config.get("output", {}).get("save_results", True):
            self._save_results()

        # Print summary
        self._print_summary()

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
                "reaction": result.decision.reaction,
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

        print("\n" + "=" * 60)
        print("COMMENTARY ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"Total dialogues: {total}")
        print(f"Reactions: {reactions} ({reactions/total*100:.1f}%)")
        print(f"Silent: {silent} ({silent/total*100:.1f}%)")
        print("=" * 60 + "\n")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Visual Novel Commentary System - Analyze VN dialogues with LLM"
    )
    parser.add_argument(
        "dialogue_file",
        help="Path to JSON file containing visual novel dialogues"
    )
    parser.add_argument(
        "--config",
        default="vn_commentary/config.yaml",
        help="Path to configuration YAML file (default: vn_commentary/config.yaml)"
    )

    args = parser.parse_args()

    # Create and run driver
    driver = VNCommentaryDriver(config_path=args.config)
    await driver.process_dialogues(args.dialogue_file)


if __name__ == "__main__":
    asyncio.run(main())
