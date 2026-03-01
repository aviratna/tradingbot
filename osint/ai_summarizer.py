"""Claude AI macro summarizer for XAUUSD OSINT intelligence.

Uses claude-haiku-4-5 (fast + cheap) with a 3-minute cache.
Non-blocking: refresh_if_needed() fires an asyncio.create_task() and
returns immediately so it never stalls the trading loop.

Returns ("", 0.5) if ANTHROPIC_API_KEY is not set or on any failure.
"""

import asyncio
import logging
import os
import time
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

_MODEL = "claude-haiku-4-5"
_CACHE_TTL = 180        # 3 minutes
_MAX_CONTEXT_CHARS = 1800
_MAX_OUTPUT_TOKENS = 200
_TEMPERATURE = 0.2      # low temperature for consistent financial analysis


def _build_system_prompt() -> str:
    return (
        "You are a concise institutional macro analyst specializing in gold (XAUUSD). "
        "Given a structured summary of recent social media narratives, sentiment data, "
        "and market events, produce a 2-3 sentence macro outlook for gold. "
        "Always end with one of: [BIAS: BULLISH], [BIAS: BEARISH], or [BIAS: NEUTRAL]. "
        "Be precise and market-focused. No fluff, no disclaimers."
    )


class AISummarizer:
    """
    Wraps Anthropic Claude for non-blocking macro summarization.

    Usage:
        summarizer = AISummarizer()
        await summarizer.refresh_if_needed(context_text)   # fire-and-forget
        summary, confidence = summarizer.get_cached_summary()  # always immediate
    """

    def __init__(self):
        self._api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        self._available = bool(self._api_key)
        self._cached_summary: str = ""
        self._cached_confidence: float = 0.5
        self._cached_at: float = 0.0
        self._refresh_task: Optional[asyncio.Task] = None
        self._refreshing: bool = False

        if self._available:
            logger.info("ai_summarizer_initialized: model=%s", _MODEL)
        else:
            logger.info("ai_summarizer_no_api_key: stub mode")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_cached_summary(self) -> Tuple[str, float]:
        """
        Returns (summary_text, confidence) immediately from cache.
        confidence is 0..1 — derived from response bias word.
        Returns ("", 0.5) if never refreshed or unavailable.
        """
        return self._cached_summary, self._cached_confidence

    async def refresh_if_needed(self, context_text: str) -> None:
        """
        Fires an async background refresh if cache is stale (>3 min).
        Returns immediately — never blocks the caller.
        Silently no-ops if already refreshing or API unavailable.
        """
        if not self._available:
            return
        if self._refreshing:
            return
        age = time.time() - self._cached_at
        if age < _CACHE_TTL:
            return
        # Launch background task
        try:
            self._refresh_task = asyncio.create_task(
                self._do_refresh(context_text)
            )
        except Exception as e:
            logger.debug("ai_summarizer_task_create_failed: %s", e)

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def cached_at(self) -> float:
        return self._cached_at

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _do_refresh(self, context_text: str) -> None:
        """Background coroutine: calls Claude API and updates cache."""
        self._refreshing = True
        try:
            summary, confidence = await self._call_claude(context_text)
            self._cached_summary = summary
            self._cached_confidence = confidence
            self._cached_at = time.time()
            logger.info(
                "ai_summarizer_refreshed: confidence=%.2f bias=%s",
                confidence,
                _extract_bias(summary),
            )
        except Exception as e:
            logger.warning("ai_summarizer_refresh_failed: %s", e)
        finally:
            self._refreshing = False

    async def _call_claude(self, context_text: str) -> Tuple[str, float]:
        """
        Actual Anthropic API call. Returns (summary_text, confidence_float).
        Raises on failure (caller catches).
        """
        try:
            import anthropic
        except ImportError:
            logger.warning("ai_summarizer: anthropic package not installed")
            return "", 0.5

        # Truncate context to avoid large token bills
        context_trimmed = context_text[:_MAX_CONTEXT_CHARS]

        client = anthropic.AsyncAnthropic(api_key=self._api_key)
        try:
            message = await client.messages.create(
                model=_MODEL,
                max_tokens=_MAX_OUTPUT_TOKENS,
                temperature=_TEMPERATURE,
                system=_build_system_prompt(),
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Current XAUUSD macro context:\n\n{context_trimmed}\n\n"
                            "Provide your 2-3 sentence gold macro outlook."
                        ),
                    }
                ],
            )
            text = message.content[0].text if message.content else ""
            confidence = _bias_to_confidence(text)
            return text.strip(), confidence
        finally:
            await client.close()

    @staticmethod
    def build_context(
        narratives: list,
        sentiment_snap,
        recent_events: list,
    ) -> str:
        """
        Build compact context string to pass to Claude.
        Designed to stay under _MAX_CONTEXT_CHARS.
        """
        lines: List[str] = []

        # Top narratives
        if narratives:
            top = narratives[:5]
            narrative_str = ", ".join(
                f"{n['name']}({n['confidence']:.0%},{n['gold_impact']})"
                for n in top
                if isinstance(n, dict)
            )
            lines.append(f"Dominant narratives: {narrative_str}")

        # Sentiment snapshot
        if sentiment_snap is not None:
            try:
                lines.append(
                    f"Sentiment: polarity={sentiment_snap.composite_polarity:+.2f}, "
                    f"fear={sentiment_snap.fear_index:.2f}, "
                    f"optimism={sentiment_snap.optimism_index:.2f}, "
                    f"score={sentiment_snap.normalized_score:.1f}/100"
                    + (" [FEAR SPIKE]" if sentiment_snap.fear_spike else "")
                )
            except Exception:
                pass

        # Recent events (last 5, max 80 chars each)
        if recent_events:
            recent = recent_events[-5:]
            for ts, msg, _ in recent:
                lines.append(f"• {msg[:80]}")

        return "\n".join(lines) if lines else "No recent market data available."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_bias(text: str) -> str:
    """Extract [BIAS: X] label from Claude response."""
    upper = text.upper()
    if "[BIAS: BULLISH]" in upper:
        return "BULLISH"
    if "[BIAS: BEARISH]" in upper:
        return "BEARISH"
    return "NEUTRAL"


def _bias_to_confidence(text: str) -> float:
    """
    Map bias label → confidence float (0..1).
    BULLISH → 0.72, BEARISH → 0.28, NEUTRAL → 0.5
    """
    bias = _extract_bias(text)
    if bias == "BULLISH":
        return 0.72
    if bias == "BEARISH":
        return 0.28
    return 0.50
