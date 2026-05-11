from importlib.metadata import PackageNotFoundError, version

from typologist.apply import apply_schema
from typologist.llm import LLM, AnthropicLLM, LLMOutputError, OpenAILLM
from typologist.typologist import Typologist

try:
    __version__ = version("typologist")
except PackageNotFoundError:  # editable install during dev before metadata is generated
    __version__ = "0.0.0+unknown"

__all__ = [
    "AnthropicLLM",
    "LLM",
    "LLMOutputError",
    "OpenAILLM",
    "Typologist",
    "__version__",
    "apply_schema",
]
