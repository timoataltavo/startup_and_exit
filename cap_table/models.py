from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict

@dataclass
class RoundSummary:
    kind: str
    name: str
    date: str
    new_money: float = 0.0
    new_shares: float = 0.0
    price_per_share: float = float("nan")
    pre_money: float = float("nan")
    post_money: float = float("nan")
    amounts_invested: Dict[str, float] = field(default_factory=dict)
    shares_received: Dict[str, float] = field(default_factory=dict)
    shares_to_vsp: Dict[str, float] = field(default_factory=dict)
    vsp_issued: Dict[str, float] = field(default_factory=dict)
