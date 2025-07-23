from enum import Enum
from typing import Dict, Optional, Any
from pydantic import BaseModel, Field, validator

class SimulationError(Exception):
    """Error related to simulation configuration or execution."""
    pass

class PositionSizing(str, Enum):
    # Either fixed size trading or fractional trading
    FIXED_SIZE = "fixed_size"
    FIXED_FRACTION = "fixed_fraction"

class SimulationConfig(BaseModel):
    initial_capital: float = Field(..., gt=0, description="Initial capital to start the simulation with")
    leverage: float = Field(default=1.0, ge=1.0, le=10.0, description="Leverage multiplier for positions")
    transaction_cost: float = Field(default=0.001, ge=0.0, le=0.05, description="Transaction cost as a fraction of the trade value")
    slippage: float = Field(default=0.0005, ge=0.0, le=0.05, description="Slippage as a fraction of trade price")
    position_sizing: PositionSizing = Field(default=PositionSizing.FIXED_FRACTION, description="Method to determine position size")
    position_size: float = Field(default=0.1, gt=0.0, description="Position size (fraction of capital or absolute units)")
    max_position_size: Optional[float] = Field(default = None, ge=0.0, description="Maximum position size (optional limit)")

    @validator('position_size')
    def validate_position_size(cls, v, values):
        if 'position_sizing' in values:
            if values['position_sizing'] == PositionSizing.FIXED_FRACTION and v>1.0:
                raise ValueError("Position size must be <= 1.0 (100%) when using fixed_fraction sizing")
        return v
    
    @validator('max_position_size')
    def validate_max_position_size(cls, v, values):
        if v is not None:
            if 'position_size' in values and v < values['position_size']:
                raise ValueError("Maximum position size must be >= position size")
        return v
    
    class Config:
        extra = "forbid"
        validate_assignment = True

        # Just an example for reference
        schema_extra = {
            "example": {
                "initial_capital": 10000.0,
                "leverage": 1.5,
                "transaction_cost": 0.001,
                "slippage": 0.0005,
                "position_sizing": "fixed_fraction",
                "position_size": 0.1
            }
        }

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SimulationConfig':
        return cls(**config_dict)