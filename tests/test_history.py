from datetime import datetime
from decimal import Decimal
import pytest
from unittest.mock import Mock, patch
from app.calculation import Calculation
from app.calculator_memento import CalculatorMemento
from app.history import LoggingObserver, AutoSaveObserver
from app.calculator import Calculator
from app.calculator_config import CalculatorConfig
# Sample setup for mock calculation
calculation_mock = Mock(spec=Calculation)
calculation_mock.operation = "addition"
calculation_mock.operand1 = 5
calculation_mock.operand2 = 3
calculation_mock.result = 8

# Test cases for LoggingObserver

@patch('logging.info')
def test_logging_observer_logs_calculation(logging_info_mock):
    observer = LoggingObserver()
    observer.update(calculation_mock)
    logging_info_mock.assert_called_once_with(
        "Calculation performed: addition (5, 3) = 8"
    )

def test_logging_observer_no_calculation():
    observer = LoggingObserver()
    with pytest.raises(AttributeError):
        observer.update(None)  # Passing None should raise an exception as there's no calculation

# Test cases for AutoSaveObserver

def test_autosave_observer_triggers_save():
    calculator_mock = Mock(spec=Calculator)
    calculator_mock.config = Mock(spec=CalculatorConfig)
    calculator_mock.config.auto_save = True
    observer = AutoSaveObserver(calculator_mock)
    
    observer.update(calculation_mock)
    calculator_mock.save_history.assert_called_once()

@patch('logging.info')
def test_autosave_observer_logs_autosave(logging_info_mock):
    calculator_mock = Mock(spec=Calculator)
    calculator_mock.config = Mock(spec=CalculatorConfig)
    calculator_mock.config.auto_save = True
    observer = AutoSaveObserver(calculator_mock)
    
    observer.update(calculation_mock)
    logging_info_mock.assert_called_once_with("History auto-saved")

def test_autosave_observer_does_not_trigger_save_when_disabled():
    calculator_mock = Mock(spec=Calculator)
    calculator_mock.config = Mock(spec=CalculatorConfig)
    calculator_mock.config.auto_save = False
    observer = AutoSaveObserver(calculator_mock)
    
    observer.update(calculation_mock)
    calculator_mock.save_history.assert_not_called()

# Additional negative test cases for AutoSaveObserver

def test_autosave_observer_invalid_calculator():
    with pytest.raises(TypeError):
        AutoSaveObserver(None)  # Passing None should raise a TypeError

def test_autosave_observer_no_calculation():
    calculator_mock = Mock(spec=Calculator)
    calculator_mock.config = Mock(spec=CalculatorConfig)
    calculator_mock.config.auto_save = True
    observer = AutoSaveObserver(calculator_mock)
    
    with pytest.raises(AttributeError):
        observer.update(None)  # Passing None should raise an exception





def test_to_dict_single_history():
    # Prepare a deterministic calculation dict
    calc_dict = {
        "operation": "Addition",
        "operand1": 2,
        "operand2": 3,
        "result": 5,
        "timestamp": "2025-10-06T17:52:10.416820"
    }

    # Create Calculation instance via from_dict to match project behavior
    calc = Calculation.from_dict(calc_dict)

    # Use a fixed timestamp for the memento
    memento_ts = datetime(2025, 10, 6, 18, 0, 0, 123456)
    memento = CalculatorMemento(history=[calc], timestamp=memento_ts)

    out = memento.to_dict()

    assert isinstance(out, dict)
    assert out["timestamp"] == memento_ts.isoformat()
    # history should serialize each Calculation via its to_dict()
    assert out["history"] == [calc.to_dict()]


def test_to_dict_empty_history():
    # Empty history should serialize to an empty list
    memento_ts = datetime(2025, 1, 1, 0, 0, 0)
    memento = CalculatorMemento(history=[], timestamp=memento_ts)

    out = memento.to_dict()

    assert isinstance(out, dict)
    assert out["timestamp"] == memento_ts.isoformat()
    assert out["history"] == []







# ...existing code...
def test_from_dict_single_history():
    calc_dict = {
        "operation": "Addition",
        "operand1": 2,
        "operand2": 3,
        "result": 5,
        "timestamp": "2025-10-06T18:00:00.123456"
    }

    ts = datetime(2025, 10, 6, 18, 0, 0, 123456)
    data = {"history": [calc_dict], "timestamp": ts.isoformat()}

    m = CalculatorMemento.from_dict(data)

    assert isinstance(m, CalculatorMemento)
    assert m.timestamp == ts
    assert len(m.history) == 1
    assert isinstance(m.history[0], Calculation)

    actual = m.history[0].to_dict()
    # compare string fields directly
    assert actual["operation"] == calc_dict["operation"]
    assert actual["timestamp"] == calc_dict["timestamp"]
    # normalize numeric fields (handles either int or numeric string)
    for key in ("operand1", "operand2", "result"):
        assert int(actual[key]) == calc_dict[key]


def test_from_dict_empty_history():
    ts = datetime(2025, 1, 1, 0, 0, 0)
    data = {"history": [], "timestamp": ts.isoformat()}

    m = CalculatorMemento.from_dict(data)

    assert isinstance(m, CalculatorMemento)
    assert m.timestamp == ts
    assert m.history == []
# ...existing code...