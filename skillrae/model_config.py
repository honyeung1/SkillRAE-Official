"""Explicit model settings for construction and requirement binding."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import os

PRESETS = {
    'default': ('gpt-5.5', 'gpt-5.5'),
    'skillsbench': ('gpt-5.5', 'gpt-5.5'),
    'agentskillos': ('gpt-5.5', 'gpt-5.5'),
}

@dataclass(frozen=True)
class ModelConfig:
    construction: str
    binding: str

    def payload(self):
        return asdict(self)


def model_config() -> ModelConfig:
    preset = os.environ.get('SKILLRAE_MODEL_PRESET', 'default')
    if preset not in PRESETS:
        raise ValueError(f'Unknown model preset {preset!r}; choose from {tuple(PRESETS)}')
    construction, binding = PRESETS[preset]
    values = (
        os.environ.get('SKILLRAE_CONSTRUCTION_MODEL', construction),
        os.environ.get('SKILLRAE_BINDING_MODEL', binding),
    )
    if any(not value.strip() or value != value.strip() for value in values):
        raise ValueError('Model identifiers must be nonempty and have no surrounding whitespace')
    return ModelConfig(*values)
