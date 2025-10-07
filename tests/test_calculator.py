import datetime
from pathlib import Path
from typing import Literal
import pandas as pd
import pytest
from unittest.mock import Mock, patch, PropertyMock
from decimal import Decimal, InvalidOperation
from tempfile import TemporaryDirectory
from app.calculation import Calculation
from app.calculator import Calculator
from app.calculator_repl import calculator_repl
from app.calculator_config import CalculatorConfig
from app.exceptions import OperationError, ValidationError
from app.history import LoggingObserver, AutoSaveObserver
from app.operations import OperationFactory
from unittest.mock import Mock, patch, MagicMock
import builtins
# Fixture to initialize Calculator with a temporary directory for file paths
@pytest.fixture
def calculator():
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        config = CalculatorConfig(base_dir=temp_path)

        # Patch properties to use the temporary directory paths
        with patch.object(CalculatorConfig, 'log_dir', new_callable=PropertyMock) as mock_log_dir, \
             patch.object(CalculatorConfig, 'log_file', new_callable=PropertyMock) as mock_log_file, \
             patch.object(CalculatorConfig, 'history_dir', new_callable=PropertyMock) as mock_history_dir, \
             patch.object(CalculatorConfig, 'history_file', new_callable=PropertyMock) as mock_history_file:
            
            # Set return values to use paths within the temporary directory
            mock_log_dir.return_value = temp_path / "logs"
            mock_log_file.return_value = temp_path / "logs/calculator.log"
            mock_history_dir.return_value = temp_path / "history"
            mock_history_file.return_value = temp_path / "history/calculator_history.csv"
            
            # Return an instance of Calculator with the mocked config
            yield Calculator(config=config)

# Test Calculator Initialization

def test_calculator_initialization(calculator: Calculator):
    assert calculator.history == []
    assert calculator.undo_stack == []
    assert calculator.redo_stack == []
    assert calculator.operation_strategy is None

# Test Logging Setup

@patch('app.calculator.logging.info')
def test_logging_setup(logging_info_mock):
    with patch.object(CalculatorConfig, 'log_dir', new_callable=PropertyMock) as mock_log_dir, \
         patch.object(CalculatorConfig, 'log_file', new_callable=PropertyMock) as mock_log_file:
        mock_log_dir.return_value = Path('/tmp/logs')
        mock_log_file.return_value = Path('/tmp/logs/calculator.log')
        
        # Instantiate calculator to trigger logging
        calculator = Calculator(CalculatorConfig())
        logging_info_mock.assert_any_call("Calculator initialized with configuration")

# Test Adding and Removing Observers

def test_add_observer(calculator: Calculator):
    observer = LoggingObserver()
    calculator.add_observer(observer)
    assert observer in calculator.observers

def test_remove_observer(calculator: Calculator):
    observer = LoggingObserver()
    calculator.add_observer(observer)
    calculator.remove_observer(observer)
    assert observer not in calculator.observers

# Test Setting Operations

def test_set_operation(calculator: Calculator):
    operation = OperationFactory.create_operation('add')
    calculator.set_operation(operation)
    assert calculator.operation_strategy == operation

# Test Performing Operations

def test_perform_operation_addition(calculator: Calculator):
    operation = OperationFactory.create_operation('add')
    calculator.set_operation(operation)
    result = calculator.perform_operation(2, 3)
    assert result == Decimal('5')

def test_perform_operation_validation_error(calculator: Calculator):
    calculator.set_operation(OperationFactory.create_operation('add'))
    with pytest.raises(ValidationError):
        calculator.perform_operation('invalid', 3)

def test_perform_operation_operation_error(calculator: Calculator):
    with pytest.raises(OperationError, match="No operation set"):
        calculator.perform_operation(2, 3)
# mypy test_calculator.py::test_division_by_zero_raises_operation_error --cov=app --cov-report=term-missing
def test_unknown_operation():
    """This covers: if not op: raise OperationError(...)"""
    calc = Calculation.__new__(Calculation)  # Avoid __post_init__
    calc.operation = "InvalidOperation"
    calc.operand1 = Decimal("10")
    calc.operand2 = Decimal("5")
    
    with pytest.raises(OperationError, match="Unknown operation: InvalidOperation"):
        calc.calculate()

# Test 2: Exception during operation execution  
def test_operation_raises_exception():
    """This covers: except (InvalidOperation, ValueError, ArithmeticError) as e:"""
    # Use patch to make pow raise an exception
    with patch('builtins.pow', side_effect=ValueError("Simulated error")):
        calc = Calculation.__new__(Calculation)
        calc.operation = "Power"  # Power uses pow() function
        calc.operand1 = Decimal("2")
        calc.operand2 = Decimal("3")
        
        with pytest.raises(OperationError, match="Calculation failed: Simulated error"):
            calc.calculate()
def test_calculation_exception():
    # TODO: Implement this test or remove if not needed
    pass
# Test Undo/Redo Functionality

def test_undo(calculator: Calculator):  
    operation = OperationFactory.create_operation('add')
    calculator.set_operation(operation)
    calculator.perform_operation(2, 3)
    calculator.undo()
    assert calculator.history == []

def test_redo(calculator: Calculator):
    operation = OperationFactory.create_operation('add')
    calculator.set_operation(operation)
    calculator.perform_operation(2, 3)
    calculator.undo()
    calculator.redo()
    assert len(calculator.history) == 1
#-----------------------------------------------------







# Test History Management
#-----------------------------------------------------
@patch('app.calculator.pd.DataFrame.to_csv')
def test_save_history(mock_to_csv, calculator: Calculator):
    operation = OperationFactory.create_operation('add')
    calculator.set_operation(operation)
    calculator.perform_operation(2, 3)
    calculator.save_history()
    mock_to_csv.assert_called_once()

@patch('app.calculator.pd.read_csv')
@patch('app.calculator.Path.exists', return_value=True)
def test_load_history(mock_exists, mock_read_csv, calculator: Calculator):
    # Mock CSV data to match the expected format in from_dict
    mock_read_csv.return_value = pd.DataFrame({
        'operation': ['Addition'],
        'operand1': ['2'],
        'operand2': ['3'],
        'result': ['5'],
        'timestamp': [datetime.datetime.now().isoformat()]
    })
    
    # Test the load_history functionality
    try:
        calculator.load_history()
        # Verify history length after loading
        assert len(calculator.history) == 1
        # Verify the loaded values
        assert calculator.history[0].operation == "Addition"
        assert calculator.history[0].operand1 == Decimal("2")
        assert calculator.history[0].operand2 == Decimal("3")
        assert calculator.history[0].result == Decimal("5")
    except OperationError:
        pytest.fail("Loading history failed due to OperationError")
        
            
# Test Clearing History

def test_clear_history(calculator: Calculator):
    operation = OperationFactory.create_operation('add')
    calculator.set_operation(operation)
    calculator.perform_operation(2, 3)
    calculator.clear_history()
    assert calculator.history == []
    assert calculator.undo_stack == []
    assert calculator.redo_stack == []

# Test REPL Commands (using patches for input/output handling)

@patch('builtins.input', side_effect=['exit'])
@patch('builtins.print')
def test_calculator_repl_exit(mock_print, mock_input):
    with patch('app.calculator.Calculator.save_history') as mock_save_history:
        calculator_repl()
        mock_save_history.assert_called_once()
        mock_print.assert_any_call("History saved successfully.")
        mock_print.assert_any_call("Goodbye!")

@patch('builtins.input', side_effect=['help', 'exit'])
@patch('builtins.print')
def test_calculator_repl_help(mock_print, mock_input):
    calculator_repl()
    mock_print.assert_any_call("\nAvailable commands:")

@patch('builtins.input', side_effect=['add', '2', '3', 'exit'])
@patch('builtins.print')
def test_calculator_repl_addition(mock_print, mock_input):
    calculator_repl()
    mock_print.assert_any_call("\nResult: 5")



#-----------------------------------------------------# Test Calculation Class

class TestExitCommand:
    """Test cases for the 'exit' command in calculator REPL."""
    
    @patch('builtins.input', side_effect=['exit'])
    @patch('builtins.print')
    def test_exit_command_saves_history_successfully(self, mock_print, mock_input):
        """Test exit command successfully saves history."""
        with patch('app.calculator_repl.Calculator') as mock_calc_class:
            # Setup mock calculator
            mock_calc = Mock()
            mock_calc_class.return_value = mock_calc
            mock_calc.save_history.return_value = None  # Success case
            
            # Run the REPL - it should return normally, not raise SystemExit
            calculator_repl()
            
            # Verify save_history was called
            mock_calc.save_history.assert_called_once()
            
            # Verify success messages were printed
            mock_print.assert_any_call("History saved successfully.")
            mock_print.assert_any_call("Goodbye!")
    
    @patch('builtins.input', side_effect=['exit'])
    @patch('builtins.print')
    def test_exit_command_handles_save_history_exception(self, mock_print, mock_input):
        """Test exit command handles save history exception gracefully."""
        with patch('app.calculator_repl.Calculator') as mock_calc_class:
            # Setup mock calculator that raises exception on save
            mock_calc = Mock()
            mock_calc_class.return_value = mock_calc
            mock_calc.save_history.side_effect = Exception("Save failed")
            
            # Run the REPL - it should return normally
            calculator_repl()
            
            # Verify save_history was called
            mock_calc.save_history.assert_called_once()
            
            # Verify warning message was printed
            mock_print.assert_any_call("Warning: Could not save history: Save failed")
            mock_print.assert_any_call("Goodbye!")
    
    @patch('builtins.input', side_effect=['help', 'exit'])
    @patch('builtins.print')
    def test_exit_command_after_other_commands(self, mock_print, mock_input):
        """Test exit command works after using other commands."""
        with patch('app.calculator_repl.Calculator') as mock_calc_class:
            # Setup mock calculator
            mock_calc = Mock()
            mock_calc_class.return_value = mock_calc
            mock_calc.save_history.return_value = None
            
            # Run the REPL
            calculator_repl()
            
            # Verify save_history was called
            mock_calc.save_history.assert_called_once()
            
            # Verify both help and exit messages were printed
            mock_print.assert_any_call("Goodbye!")
    
    @patch('builtins.input', side_effect=['exit'])
    @patch('builtins.print')
    def test_exit_command_with_specific_exception_types(self, mock_print, mock_input):
        """Test exit command handles different exception types."""
        with patch('app.calculator_repl.Calculator') as mock_calc_class:
            # Setup mock calculator with different exception types
            mock_calc = Mock()
            mock_calc_class.return_value = mock_calc
            
            # Test with IOError
            mock_calc.save_history.side_effect = IOError("File not found")
            
            calculator_repl()
            
            mock_print.assert_any_call("Warning: Could not save history: File not found")
    
    @patch('builtins.input', side_effect=['exit'])
    @patch('builtins.print')
    def test_exit_command_break_loop(self, mock_print, mock_input):
        """Test that exit command breaks the REPL loop."""
        with patch('app.calculator_repl.Calculator') as mock_calc_class:
            # Setup mock calculator
            mock_calc = Mock()
            mock_calc_class.return_value = mock_calc
            mock_calc.save_history.return_value = None
            
            # The REPL should exit without further input
            call_count_before = mock_input.call_count
            
            calculator_repl()
            
            # Input should only be called once (for 'exit')
            assert mock_input.call_count == call_count_before + 1
    
    @patch('builtins.input', side_effect=['exit'])
    @patch('builtins.print')
    def test_exit_command_observers_registered(self, mock_print, mock_input):
        """Test that observers are registered before exit command."""
        with patch('app.calculator_repl.Calculator') as mock_calc_class:
            with patch('app.calculator_repl.LoggingObserver') as mock_logging_obs:
                with patch('app.calculator_repl.AutoSaveObserver') as mock_auto_save_obs:
                    # Setup mocks
                    mock_calc = Mock()
                    mock_calc_class.return_value = mock_calc
                    mock_logging_obs.return_value = Mock()
                    mock_auto_save_obs.return_value = Mock()
                    
                    # Run REPL
                    calculator_repl()
                    
                    # Verify observers were added
                    assert mock_calc.add_observer.call_count == 2
                    mock_calc.save_history.assert_called_once()
    
    @patch('builtins.input', side_effect=['exit'])
    @patch('builtins.print')
    def test_exit_command_only_called_once(self, mock_print, mock_input):
        """Test that exit command only processes once and doesn't loop."""
        with patch('app.calculator_repl.Calculator') as mock_calc_class:
            mock_calc = Mock()
            mock_calc_class.return_value = mock_calc
            mock_calc.save_history.return_value = None
            
            # This should not loop - exit should break immediately
            calculator_repl()
            
            # Input should be called exactly once
            assert mock_input.call_count == 1
            mock_calc.save_history.assert_called_once()



class TestUndoCommand:
    """Test cases for the 'undo' command in calculator REPL."""
    
    @patch('builtins.input', side_effect=['undo', 'exit'])
    @patch('builtins.print')
    def test_undo_command_successful(self, mock_print, mock_input):
        """Test undo command when there is something to undo."""
        with patch('app.calculator_repl.Calculator') as mock_calc_class:
            # Setup mock calculator with successful undo
            mock_calc = Mock()
            mock_calc_class.return_value = mock_calc
            mock_calc.undo.return_value = True  # Success case
            
            calculator_repl()
            
            # Verify undo was called
            mock_calc.undo.assert_called_once()
            
            # Verify success message was printed
            mock_print.assert_any_call("Operation undone")
    
    @patch('builtins.input', side_effect=['undo', 'exit'])
    @patch('builtins.print')
    def test_undo_command_nothing_to_undo(self, mock_print, mock_input):
        """Test undo command when there is nothing to undo."""
        with patch('app.calculator_repl.Calculator') as mock_calc_class:
            # Setup mock calculator with nothing to undo
            mock_calc = Mock()
            mock_calc_class.return_value = mock_calc
            mock_calc.undo.return_value = False  # Nothing to undo
            
            calculator_repl()
            
            # Verify undo was called
            mock_calc.undo.assert_called_once()
            
            # Verify nothing to undo message was printed
            mock_print.assert_any_call("Nothing to undo")
    
    @patch('builtins.input', side_effect=['undo', 'undo', 'exit'])
    @patch('builtins.print')
    def test_undo_command_multiple_calls(self, mock_print, mock_input):
        """Test multiple undo commands in sequence."""
        with patch('app.calculator_repl.Calculator') as mock_calc_class:
            # Setup mock calculator with mixed undo results
            mock_calc = Mock()
            mock_calc_class.return_value = mock_calc
            mock_calc.undo.side_effect = [True, False]  # First succeeds, second fails
            
            calculator_repl()
            
            # Verify undo was called twice
            assert mock_calc.undo.call_count == 2
            
            # Verify both messages were printed
            mock_print.assert_any_call("Operation undone")
            mock_print.assert_any_call("Nothing to undo")
    
    @patch('builtins.input', side_effect=['add', '5', '3', 'undo', 'exit'])
    @patch('builtins.print')
    def test_undo_after_calculation(self, mock_print, mock_input):
        """Test undo command after performing a calculation."""
        with patch('app.calculator_repl.Calculator') as mock_calc_class:
            with patch('app.calculator_repl.OperationFactory') as mock_factory:
                # Setup mocks
                mock_calc = Mock()
                mock_calc_class.return_value = mock_calc
                mock_calc.undo.return_value = True
                mock_calc.perform_operation.return_value = 8
                
                mock_operation = Mock()
                mock_factory.create_operation.return_value = mock_operation
                
                calculator_repl()
                
                # Verify undo was called after calculation
                mock_calc.undo.assert_called_once()
                mock_print.assert_any_call("Operation undone")
    
    @patch('builtins.input', side_effect=['undo', 'history', 'exit'])
    @patch('builtins.print')
    def test_undo_before_history(self, mock_print, mock_input):
        """Test undo command followed by history command."""
        with patch('app.calculator_repl.Calculator') as mock_calc_class:
            # Setup mock calculator
            mock_calc = Mock()
            mock_calc_class.return_value = mock_calc
            mock_calc.undo.return_value = True
            mock_calc.show_history.return_value = []  # Empty history
            
            calculator_repl()
            
            # Verify undo was called
            mock_calc.undo.assert_called_once()
            mock_print.assert_any_call("Operation undone")
    
    @patch('builtins.input', side_effect=['undo', 'redo', 'exit'])
    @patch('builtins.print')
    def test_undo_followed_by_redo(self, mock_print, mock_input):
        """Test undo command followed by redo command."""
        with patch('app.calculator_repl.Calculator') as mock_calc_class:
            # Setup mock calculator
            mock_calc = Mock()
            mock_calc_class.return_value = mock_calc
            mock_calc.undo.return_value = True
            mock_calc.redo.return_value = True
            
            calculator_repl()
            
            # Verify both undo and redo were called
            mock_calc.undo.assert_called_once()
            mock_calc.redo.assert_called_once()
            mock_print.assert_any_call("Operation undone")
            mock_print.assert_any_call("Operation redone")




@pytest.mark.parametrize("command,method_name,result,expected_message", [
    ('redo', 'redo', True, "Operation redone"),
    ('redo', 'redo', False, "Nothing to redo"),
])
@patch('builtins.input')
@patch('builtins.print')
def test_basic_commands(mock_print, mock_input, command: builtins.str, method_name: builtins.str, result: builtins.bool, expected_message: builtins.str | builtins.str):
    """Test basic commands with different outcomes."""
    mock_input.side_effect = [command, 'exit']
    
    with patch('app.calculator_repl.Calculator') as mock_calc_class:
        mock_calc = Mock()
        mock_calc_class.return_value = mock_calc
        
        method = getattr(mock_calc, method_name)
        method.return_value = result
        
        calculator_repl()
        
        method.assert_called_once()
        mock_print.assert_any_call(expected_message)


@patch('builtins.input')
@patch('builtins.print')
def test_save_command(mock_print, mock_input):
    """Test save command separately to handle double call."""
    mock_input.side_effect = ['save', 'exit']
    
    with patch('app.calculator_repl.Calculator') as mock_calc_class:
        mock_calc = Mock()
        mock_calc_class.return_value = mock_calc
        
        calculator_repl()
        
        # save_history is called twice: once by 'save' command, once by 'exit' command
        assert mock_calc.save_history.call_count == 2
        mock_print.assert_any_call("History saved successfully")


@patch('builtins.input')
@patch('builtins.print')
def test_save_command_with_error(mock_print, mock_input):
    """Test save command error handling."""
    mock_input.side_effect = ['save', 'exit']
    
    with patch('app.calculator_repl.Calculator') as mock_calc_class:
        mock_calc = Mock()
        mock_calc_class.return_value = mock_calc
        
        # First call (save command) fails, second call (exit) succeeds
        mock_calc.save_history.side_effect = [
            Exception("Save failed"),  # First call - save command
            None  # Second call - exit command
        ]
        
        calculator_repl()
        
        # Both calls should still happen
        assert mock_calc.save_history.call_count == 2
        mock_print.assert_any_call("Error saving history: Save failed")
        mock_print.assert_any_call("History saved successfully.")  # From exit


@patch('builtins.input', side_effect=['undo', 'redo', 'exit'])
@patch('builtins.print')
def test_complete_sequence(mock_print, mock_input):
    """Test a complete command sequence without save to avoid double calls."""
    with patch('app.calculator_repl.Calculator') as mock_calc_class:
        mock_calc = Mock()
        mock_calc_class.return_value = mock_calc
        mock_calc.undo.return_value = True
        mock_calc.redo.return_value = True
        
        calculator_repl()
        
        # Verify all commands were processed
        mock_calc.undo.assert_called_once()
        mock_calc.redo.assert_called_once()
        mock_calc.save_history.assert_called_once()  # Only from exit command
        
        # Verify all success messages
        mock_print.assert_any_call("Operation undone")
        mock_print.assert_any_call("Operation redone")
        mock_print.assert_any_call("History saved successfully.")  # From exit


@patch('builtins.input', side_effect=['save', 'save', 'exit'])
@patch('builtins.print')
def test_multiple_save_commands(mock_print, mock_input):
    """Test multiple save commands."""
    with patch('app.calculator_repl.Calculator') as mock_calc_class:
        mock_calc = Mock()
        mock_calc_class.return_value = mock_calc
        
        calculator_repl()
        
        # Two explicit saves + one auto-save on exit = 3 total
        assert mock_calc.save_history.call_count == 3
        
        # Should see success message for each explicit save
        save_success_calls = [call for call in mock_print.call_args_list 
                            if call[0][0] == "History saved successfully"]
        assert len(save_success_calls) == 2  # Two explicit saves\\

        
@patch('builtins.input')
@patch('builtins.print')
def test_unknown_command(mock_print, mock_input):
    """Test unknown command handling."""
    mock_input.side_effect = ['invalid', 'exit']
    with patch('app.calculator_repl.Calculator'):
        calculator_repl()
        mock_print.assert_any_call("Unknown command: 'invalid'. Type 'help' for available commands.")


@patch('builtins.input')
@patch('builtins.print')
def test_keyboard_interrupt(mock_print, mock_input):
    """Test Ctrl+C handling."""
    mock_input.side_effect = [KeyboardInterrupt, 'exit']
    with patch('app.calculator_repl.Calculator'):
        calculator_repl()
        mock_print.assert_any_call("\nOperation cancelled")


@patch('builtins.input')
@patch('builtins.print')
def test_eof_error(mock_print, mock_input):
    """Test Ctrl+D handling."""
    mock_input.side_effect = [EOFError]
    with patch('app.calculator_repl.Calculator'):
        calculator_repl()
        mock_print.assert_any_call("\nInput terminated. Exiting...")


@patch('builtins.input')
@patch('builtins.print')
def test_general_exception(mock_print, mock_input):
    """Test general exception handling."""
    mock_input.side_effect = [Exception("Test error"), 'exit']
    with patch('app.calculator_repl.Calculator'):
        calculator_repl()
        mock_print.assert_any_call("Error: Test error")


@patch('builtins.input', side_effect=['exit'])
@patch('builtins.print')
@patch('app.calculator_repl.logging')
def test_fatal_error(mock_logging, mock_print, mock_input):
    """Test fatal initialization error."""
    with patch('app.calculator_repl.Calculator') as mock_calc:
        mock_calc.side_effect = Exception("Init failed")
        with pytest.raises(Exception):
            calculator_repl()
        mock_print.assert_any_call("Fatal error: Init failed")
        mock_logging.error.assert_called_once()







import app.calculator_repl as repl_module



def _mock_calculator_factory(perform_side_effect=None):
    """
    Return a callable that produces a mock Calculator instance.
    The instance implements the methods used by calculator_repl().
    """
    def _factory():
        inst = Mock()
        inst.add_observer = Mock()
        inst.save_history = Mock()
        inst.set_operation = Mock()
        inst.perform_operation = Mock(side_effect=perform_side_effect)
        inst.show_history = Mock(return_value=[])
        inst.clear_history = Mock()
        inst.undo = Mock(return_value=False)
        inst.redo = Mock(return_value=False)
        return inst
    return _factory


def test_repl_handles_validation_error(monkeypatch, capsys):
    # Calculator.perform_operation will raise a ValidationError
    monkeypatch.setattr(repl_module, "Calculator", _mock_calculator_factory(perform_side_effect=ValidationError("bad input")))

    inputs = iter(["add", "1", "2", "exit"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(inputs))

    repl_module.calculator_repl()

    out = capsys.readouterr().out
    assert "Error: bad input" in out


def test_repl_handles_unexpected_exception(monkeypatch, capsys):
    # Calculator.perform_operation will raise a generic Exception
    monkeypatch.setattr(repl_module, "Calculator", _mock_calculator_factory(perform_side_effect=Exception("boom")))

    inputs = iter(["add", "1", "2", "exit"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(inputs))

    repl_module.calculator_repl()

    out = capsys.readouterr().out
    assert "Unexpected error: boom" in out








