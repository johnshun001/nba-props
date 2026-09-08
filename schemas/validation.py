"""Pydantic validation schemas for NBA prop betting system."""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, field_validator, model_validator


class PropLine(BaseModel):
    """
    Schema for a player prop line from a sportsbook.

    Validates all API responses containing prop line data.
    """
    player_id: str = Field(..., description="Unique player identifier")
    prop_type: str = Field(..., description="Type of prop (e.g., 'points', 'rebounds', 'assists')")
    line: float = Field(..., gt=0, description="The prop line value")
    over_odds: int = Field(..., ge=-1000, le=1000, description="American odds for over bet")
    under_odds: int = Field(..., ge=-1000, le=1000, description="American odds for under bet")
    sportsbook: str = Field(..., min_length=1, description="Sportsbook name")

    @field_validator('prop_type')
    @classmethod
    def validate_prop_type(cls, v: str) -> str:
        """Validate prop type is one of allowed values."""
        allowed = {'points', 'rebounds', 'assists', 'threes', 'steals', 'blocks',
                   'turnovers', 'points_rebounds_assists', 'points_rebounds',
                   'points_assists', 'rebounds_assists'}
        v_lower = v.lower()
        if v_lower not in allowed:
            raise ValueError(f"Invalid prop_type '{v}'. Must be one of: {allowed}")
        return v_lower

    @field_validator('sportsbook')
    @classmethod
    def validate_sportsbook(cls, v: str) -> str:
        """Validate sportsbook name."""
        allowed = {'draftkings', 'fanduel', 'betmgm', 'caesars', 'pointsbet',
                   'betrivers', 'unibet', 'wynnbet', 'barstool', 'hardrock'}
        v_lower = v.lower()
        if v_lower not in allowed:
            raise ValueError(f"Invalid sportsbook '{v}'. Must be one of: {allowed}")
        return v_lower

    @model_validator(mode='after')
    def validate_odds_consistency(self) -> 'PropLine':
        """Validate that odds are within reasonable ranges."""
        # Both odds shouldn't be heavily negative (indicates bad line)
        if self.over_odds < -200 and self.under_odds < -200:
            raise ValueError("Both over and under odds cannot be less than -200")
        return self


class Bet(BaseModel):
    """
    Schema for a placed bet.

    Validates all bet outputs before logging to database.
    """
    player_id: str = Field(..., description="Unique player identifier")
    prop_type: str = Field(..., description="Type of prop bet")
    line: float = Field(..., gt=0, description="The prop line value")
    bet_size: float = Field(..., gt=0, le=1000, description="Bet size in dollars")
    edge: float = Field(..., ge=-1, le=1, description="Model edge (0.05 = 5% edge)")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Bet timestamp")
    bet_side: str = Field(..., description="Bet side: 'over' or 'under'")
    sportsbook: str = Field(..., description="Sportsbook where bet was placed")
    odds: int = Field(..., ge=-1000, le=1000, description="American odds of the bet")

    @field_validator('prop_type')
    @classmethod
    def validate_prop_type(cls, v: str) -> str:
        """Validate prop type is one of allowed values."""
        allowed = {'points', 'rebounds', 'assists', 'threes', 'steals', 'blocks',
                   'turnovers', 'points_rebounds_assists', 'points_rebounds',
                   'points_assists', 'rebounds_assists'}
        v_lower = v.lower()
        if v_lower not in allowed:
            raise ValueError(f"Invalid prop_type '{v}'. Must be one of: {allowed}")
        return v_lower

    @field_validator('bet_side')
    @classmethod
    def validate_bet_side(cls, v: str) -> str:
        """Validate bet side."""
        v_lower = v.lower()
        if v_lower not in {'over', 'under'}:
            raise ValueError(f"Invalid bet_side '{v}'. Must be 'over' or 'under'")
        return v_lower

    @field_validator('edge')
    @classmethod
    def validate_edge(cls, v: float) -> float:
        """Validate edge is positive (we only bet with edge)."""
        if v <= 0:
            raise ValueError("Edge must be positive to place a bet")
        return v

    @model_validator(mode='after')
    def validate_bet_consistency(self) -> 'Bet':
        """Validate bet consistency."""
        # Bet size should be reasonable relative to edge
        max_bet_for_edge = self.edge * 1000  # $10 per 1% edge
        if self.bet_size > max_bet_for_edge * 2:
            raise ValueError(f"Bet size ${self.bet_size} too large for {self.edge:.1%} edge")
        return self


class BetResult(BaseModel):
    """
    Schema for a settled bet result.
    """
    bet: Bet
    outcome: str = Field(..., description="Bet outcome: 'win', 'loss', 'push'")
    pnl: float = Field(..., description="Profit and loss in dollars")
    settled_at: datetime = Field(default_factory=datetime.utcnow, description="Settlement timestamp")

    @field_validator('outcome')
    @classmethod
    def validate_outcome(cls, v: str) -> str:
        """Validate outcome."""
        v_lower = v.lower()
        if v_lower not in {'win', 'loss', 'push'}:
            raise ValueError(f"Invalid outcome '{v}'. Must be 'win', 'loss', or 'push'")
        return v_lower

    @model_validator(mode='after')
    def validate_pnl(self) -> 'BetResult':
        """Validate P&L matches outcome."""
        bet = self.bet

        if self.outcome == 'win':
            # Win: profit = bet_size * (odds/100) for positive, bet_size for negative
            if bet.odds > 0:
                expected_pnl = bet.bet_size * (bet.odds / 100)
            else:
                expected_pnl = bet.bet_size * (100 / abs(bet.odds))
            if abs(self.pnl - expected_pnl) > 0.01:
                raise ValueError(f"P&L ${self.pnl} doesn't match expected win ${expected_pnl}")
        elif self.outcome == 'loss':
            if abs(self.pnl + bet.bet_size) > 0.01:
                raise ValueError(f"P&L ${self.pnl} doesn't match expected loss -${bet.bet_size}")
        elif self.outcome == 'push':
            if abs(self.pnl) > 0.01:
                raise ValueError(f"P&L ${self.pnl} should be 0 for push")

        return self


class RevenueAttribution(BaseModel):
    """
    Schema for revenue attribution per bet.

    Tracks value from each component:
    - line_shopping_value: Value gained from selecting best line
    - kelly_edge: Value from Kelly sizing optimization
    - model_edge: Value from model prediction accuracy
    - matchup_edge: Value from matchup-specific adjustments
    """
    bet_id: str = Field(..., description="Unique bet identifier")
    line_shopping_value: float = Field(..., ge=-10, le=10, description="Value from line shopping in points")
    kelly_edge: float = Field(..., ge=-0.5, le=0.5, description="Edge from Kelly optimization")
    model_edge: float = Field(..., ge=-0.5, le=0.5, description="Edge from model prediction")
    matchup_edge: float = Field(..., ge=-0.5, le=0.5, description="Edge from matchup analysis")
    total_edge: float = Field(..., description="Sum of all edge components")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @model_validator(mode='after')
    def validate_total_edge(self) -> 'RevenueAttribution':
        """Validate total edge equals sum of components."""
        expected_total = self.line_shopping_value / 10 + self.kelly_edge + self.model_edge + self.matchup_edge
        if abs(self.total_edge - expected_total) > 0.01:
            raise ValueError(f"Total edge {self.total_edge} doesn't match sum of components {expected_total}")
        return self


class ValidationResponse(BaseModel):
    """
    Schema for API response validation results.
    """
    valid: bool = Field(..., description="Whether validation passed")
    errors: List[str] = Field(default_factory=list, description="List of validation errors")
    warnings: List[str] = Field(default_factory=list, description="List of validation warnings")
    data: Optional[dict] = Field(default=None, description="Validated data if successful")


def validate_prop_line(data: dict) -> ValidationResponse:
    """Validate a prop line dictionary."""
    try:
        prop_line = PropLine(**data)
        return ValidationResponse(valid=True, data=prop_line.model_dump())
    except Exception as e:
        return ValidationResponse(valid=False, errors=[str(e)])


def validate_bet(data: dict) -> ValidationResponse:
    """Validate a bet dictionary."""
    try:
        bet = Bet(**data)
        return ValidationResponse(valid=True, data=bet.model_dump())
    except Exception as e:
        return ValidationResponse(valid=False, errors=[str(e)])


def validate_bet_result(data: dict) -> ValidationResponse:
    """Validate a bet result dictionary."""
    try:
        bet_result = BetResult(**data)
        return ValidationResponse(valid=True, data=bet_result.model_dump())
    except Exception as e:
        return ValidationResponse(valid=False, errors=[str(e)])
