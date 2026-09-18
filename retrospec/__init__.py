"""RetroSpec v3 - a correct, measurable comparative study of inference-time
decoding strategies."""
__version__ = "3.0.0"

from .engine import (GenResult, TorchVerifier, Verifier,
                     greedy_generate, speculative_generate)

__all__ = ["GenResult", "TorchVerifier", "Verifier",
           "greedy_generate", "speculative_generate"]
