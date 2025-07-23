import pytest
from pydantic import ValidationError
from src.backtester.simulation.config import SimulationConfig, PositionSizing, SimulationError

class TestSimulationConfig:
    def test_valid_config(self):
        config = SimulationConfig(initial_capital=10000.0)
        assert config.initial_capital == 10000.0
        assert config.leverage == 1.0
        assert config.transaction_cost == 0.001
        assert config.slippage == 0.0005
        assert config.position_sizing == PositionSizing.FIXED_FRACTION
        assert config.position_size == 0.1
        assert config.max_position_size is None
        
        config = SimulationConfig(initial_capital=5000.0, leverage=2.0, transaction_cost=0.002, slippage=0.001, position_sizing=PositionSizing.FIXED_SIZE, position_size=5.0, max_position_size=10.0)
        assert config.initial_capital == 5000.0
        assert config.leverage == 2.0
        assert config.transaction_cost == 0.002
        assert config.slippage == 0.001
        assert config.position_sizing == PositionSizing.FIXED_SIZE
        assert config.position_size == 5.0
        assert config.max_position_size == 10.0
    
    def test_invalid_initial_capital(self):
        with pytest.raises(ValidationError, match="greater than 0"): # call the vcs lol
            SimulationConfig(initial_capital=0.0)
        
        with pytest.raises(ValidationError, match="greater than 0"):
            SimulationConfig(initial_capital=-1000.0)
    
    def test_invalid_leverage(self):
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            SimulationConfig(initial_capital=10000.0, leverage=0.5)
        
        with pytest.raises(ValidationError, match="less than or equal to 10"):
            SimulationConfig(initial_capital=10000.0, leverage=11.0)
    
    def test_invalid_transaction_cost(self):
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            SimulationConfig(initial_capital=10000.0, transaction_cost=-0.001)
        
        with pytest.raises(ValidationError, match="less than or equal to 0.05"):
            SimulationConfig(initial_capital=10000.0, transaction_cost=0.06)
    
    def test_invalid_slippage(self):
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            SimulationConfig(initial_capital=10000.0, slippage=-0.001)
        
        with pytest.raises(ValidationError, match="less than or equal to 0.05"):
            SimulationConfig(initial_capital=10000.0, slippage=0.06)
    
    def test_invalid_position_sizing(self):
        with pytest.raises(ValidationError, match="Input should be 'fixed_size' or 'fixed_fraction'"):
            SimulationConfig(initial_capital=10000.0, position_sizing="invalid")
    
    def test_invalid_position_size(self):
        with pytest.raises(ValidationError, match="greater than 0"):
            SimulationConfig(initial_capital=10000.0, position_size=0.0)
        
        with pytest.raises(ValidationError, match="Position size must be <= 1.0"):
            SimulationConfig(initial_capital=10000.0, position_sizing=PositionSizing.FIXED_FRACTION, position_size=1.5)
        
        config = SimulationConfig(initial_capital=10000.0, position_sizing=PositionSizing.FIXED_SIZE, position_size=5.0)
        assert config.position_size == 5.0
    
    def test_invalid_max_position_size(self):
        with pytest.raises(ValidationError, match="Maximum position size must be >= position size"):
            SimulationConfig(initial_capital=10000.0, position_size=0.2, max_position_size=0.1)
    
    def test_to_dict_method(self):
        config = SimulationConfig(initial_capital=10000.0)
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["initial_capital"] == 10000.0
        assert config_dict["leverage"] == 1.0
        assert config_dict["position_sizing"] == "fixed_fraction"
    
    def test_from_dict_method(self):
        config_dict = {
            "initial_capital": 5000.0,
            "leverage": 2.0,
            "transaction_cost": 0.002
        }
        config = SimulationConfig.from_dict(config_dict)
        assert config.initial_capital == 5000.0
        assert config.leverage == 2.0
        assert config.transaction_cost == 0.002
        assert config.slippage == 0.0005  # Default value
    
    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            SimulationConfig(
                initial_capital=10000.0,
                unknown_field="value"
            )