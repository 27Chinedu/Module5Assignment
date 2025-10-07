import pytest
from decimal import Decimal
from datetime import datetime
from app.calculation import Calculation
from app.exceptions import OperationError
import logging


def test_addition():
    calc = Calculation(operation="Addition", operand1=Decimal("2"), operand2=Decimal("3"))
    assert calc.result == Decimal("5")


def test_subtraction():
    calc = Calculation(operation="Subtraction", operand1=Decimal("5"), operand2=Decimal("3"))
    assert calc.result == Decimal("2")


def test_multiplication():
    calc = Calculation(operation="Multiplication", operand1=Decimal("4"), operand2=Decimal("2"))
    assert calc.result == Decimal("8")


def test_division():
    calc = Calculation(operation="Division", operand1=Decimal("8"), operand2=Decimal("2"))
    assert calc.result == Decimal("4")


def test_division_by_zero():
    with pytest.raises(OperationError, match="Division by zero is not allowed"):
        Calculation(operation="Division", operand1=Decimal("8"), operand2=Decimal("0"))


def test_power():
    calc = Calculation(operation="Power", operand1=Decimal("2"), operand2=Decimal("3"))
    assert calc.result == Decimal("8")


def test_negative_power():
    with pytest.raises(OperationError, match="Negative exponents are not supported"):
        Calculation(operation="Power", operand1=Decimal("2"), operand2=Decimal("-3"))


def test_root():
    calc = Calculation(operation="Root", operand1=Decimal("16"), operand2=Decimal("2"))
    assert calc.result == Decimal("4")


def test_invalid_root():
    with pytest.raises(OperationError, match="Cannot calculate root of negative number"):
        Calculation(operation="Root", operand1=Decimal("-16"), operand2=Decimal("2"))


def test_unknown_operation():
    with pytest.raises(OperationError, match="Unknown operation"):
        Calculation(operation="Unknown", operand1=Decimal("5"), operand2=Decimal("3"))


def test_to_dict():
    calc = Calculation(operation="Addition", operand1=Decimal("2"), operand2=Decimal("3"))
    result_dict = calc.to_dict()
    assert result_dict == {
        "operation": "Addition",
        "operand1": "2",
        "operand2": "3",
        "result": "5",
        "timestamp": calc.timestamp.isoformat()
    }


def test_from_dict():
    data = {
        "operation": "Addition",
        "operand1": "2",
        "operand2": "3",
        "result": "5",
        "timestamp": datetime.now().isoformat()
    }
    calc = Calculation.from_dict(data)
    assert calc.operation == "Addition"
    assert calc.operand1 == Decimal("2")
    assert calc.operand2 == Decimal("3")
    assert calc.result == Decimal("5")


def test_invalid_from_dict():
    data = {
        "operation": "Addition",
        "operand1": "invalid",
        "operand2": "3",
        "result": "5",
        "timestamp": datetime.now().isoformat()
    }
    with pytest.raises(OperationError, match="Invalid calculation data"):
        Calculation.from_dict(data)


def test_format_result():
    calc = Calculation(operation="Division", operand1=Decimal("1"), operand2=Decimal("3"))
    assert calc.format_result(precision=2) == "0.33"
    assert calc.format_result(precision=10) == "0.3333333333"


def test_equality():
    calc1 = Calculation(operation="Addition", operand1=Decimal("2"), operand2=Decimal("3"))
    calc2 = Calculation(operation="Addition", operand1=Decimal("2"), operand2=Decimal("3"))
    calc3 = Calculation(operation="Subtraction", operand1=Decimal("5"), operand2=Decimal("3"))
    assert calc1 == calc2
    assert calc1 != calc3


# New Test to Cover Logging Warning
def test_from_dict_result_mismatch(caplog):
    """
    Test the from_dict method to ensure it logs a warning when the saved result
    does not match the computed result.
    """
    # Arrange
    data = {
        "operation": "Addition",
        "operand1": "2",
        "operand2": "3",
        "result": "10",  # Incorrect result to trigger logging.warning
        "timestamp": datetime.now().isoformat()
    }

    # Act
    with caplog.at_level(logging.WARNING):
        calc = Calculation.from_dict(data)

    # Assert
    assert "Loaded calculation result 10 differs from computed result 5" in caplog.text
#---------------------------------------------------------------------
#Chinedu Erechukwu




class TestCalculationStringRepresentation:
    """Test cases for Calculation string representation methods."""
    
    def test_str_representation_addition(self):
        """Test __str__ method with addition operation."""
        calc = Calculation('Addition', Decimal('5'), Decimal('3'))
        expected = "Addition(5, 3) = 8"
        assert str(calc) == expected
    
    def test_str_representation_subtraction(self):
        """Test __str__ method with subtraction operation."""
        calc = Calculation('Subtraction', Decimal('10'), Decimal('4'))
        expected = "Subtraction(10, 4) = 6"
        assert str(calc) == expected
    
    def test_str_representation_multiplication(self):
        """Test __str__ method with multiplication operation."""
        calc = Calculation('Multiplication', Decimal('7'), Decimal('2'))
        expected = "Multiplication(7, 2) = 14"
        assert str(calc) == expected
    
    def test_str_representation_division(self):
        """Test __str__ method with division operation."""
        calc = Calculation('Division', Decimal('15'), Decimal('3'))
        expected = "Division(15, 3) = 5"
        assert str(calc) == expected
    
    def test_str_representation_power(self):
        """Test __str__ method with power operation."""
        calc = Calculation('Power', Decimal('2'), Decimal('3'))
        expected = "Power(2, 3) = 8"
        assert str(calc) == expected
    
    def test_str_representation_root(self):
        """Test __str__ method with root operation."""
        calc = Calculation('Root', Decimal('8'), Decimal('3'))
        expected = "Root(8, 3) = 2"
        assert str(calc) == expected
    
    def test_str_representation_with_decimal_operands(self):
        """Test __str__ method with decimal operands."""
        calc = Calculation('Addition', Decimal('2.5'), Decimal('3.75'))
        expected = "Addition(2.5, 3.75) = 6.25"
        assert str(calc) == expected
    
    def test_str_representation_with_negative_operands(self):
        """Test __str__ method with negative operands."""
        calc = Calculation('Subtraction', Decimal('-5'), Decimal('-3'))
        expected = "Subtraction(-5, -3) = -2"
        assert str(calc) == expected
    
    def test_str_representation_with_zero_operands(self):
        """Test __str__ method with zero operands."""
        calc = Calculation('Multiplication', Decimal('0'), Decimal('5'))
        expected = "Multiplication(0, 5) = 0"
        assert str(calc) == expected
    
    def test_str_representation_large_numbers(self):
        """Test __str__ method with large numbers."""
        calc = Calculation('Addition', Decimal('1000000'), Decimal('2000000'))
        expected = "Addition(1000000, 2000000) = 3000000"
        assert str(calc) == expected
    
    def test_repr_representation(self):
        """Test __repr__ method for completeness."""
        calc = Calculation('Addition', Decimal('5'), Decimal('3'))
        repr_str = repr(calc)
        
        # Check that all important attributes are in the repr
        assert "Calculation(" in repr_str
        assert "operation='Addition'" in repr_str
        assert "operand1=5" in repr_str
        assert "operand2=3" in repr_str
        assert "result=8" in repr_str
        assert "timestamp=" in repr_str
    
    def test_str_vs_repr_difference(self):
        """Test that __str__ and __repr__ return different formats."""
        calc = Calculation('Addition', Decimal('5'), Decimal('3'))
        
        str_result = str(calc)
        repr_result = repr(calc)
        
        # __str__ should be more human-readable
        assert str_result == "Addition(5, 3) = 8"
        
        # __repr__ should be more detailed and include class name
        assert "Calculation(" in repr_result
        assert "operation='Addition'" in repr_result



class TestCalculationEquality:
    """Test cases for Calculation equality method."""
    
    def test_eq_same_calculations(self):
        """Test that two identical calculations are equal."""
        calc1 = Calculation('Addition', Decimal('5'), Decimal('3'))
        calc2 = Calculation('Addition', Decimal('5'), Decimal('3'))
        
        assert calc1 == calc2
        assert not (calc1 != calc2)
    
    def test_eq_different_operation(self):
        """Test that calculations with different operations are not equal."""
        calc1 = Calculation('Addition', Decimal('5'), Decimal('3'))
        calc2 = Calculation('Subtraction', Decimal('5'), Decimal('3'))
        
        assert calc1 != calc2
        assert not (calc1 == calc2)
    
    def test_eq_different_operand1(self):
        """Test that calculations with different first operands are not equal."""
        calc1 = Calculation('Addition', Decimal('5'), Decimal('3'))
        calc2 = Calculation('Addition', Decimal('10'), Decimal('3'))
        
        assert calc1 != calc2
        assert not (calc1 == calc2)
    
    def test_eq_different_operand2(self):
        """Test that calculations with different second operands are not equal."""
        calc1 = Calculation('Addition', Decimal('5'), Decimal('3'))
        calc2 = Calculation('Addition', Decimal('5'), Decimal('7'))
        
        assert calc1 != calc2
        assert not (calc1 == calc2)
    
    def test_eq_different_result(self):
        """Test that calculations with different results are not equal."""
        # Create two calculations that should have different results
        calc1 = Calculation('Addition', Decimal('5'), Decimal('3'))  # Result: 8
        calc2 = Calculation('Multiplication', Decimal('2'), Decimal('3'))  # Result: 6
        
        # Force different result while keeping other attributes the same
        calc2.operation = 'Addition'
        calc2.operand1 = Decimal('5')
        calc2.operand2 = Decimal('3')
        # calc2.result is still 6 from the original calculation
        
        assert calc1 != calc2
        assert not (calc1 == calc2)
    
    def test_eq_same_operation_different_types(self):
        """Test that calculations with same values but different decimal types are equal."""
        calc1 = Calculation('Addition', Decimal('5.0'), Decimal('3.00'))
        calc2 = Calculation('Addition', Decimal('5'), Decimal('3'))
        
        # Decimals with different precision but same value should be equal
        assert calc1 == calc2
    
    def test_eq_with_different_object_type(self):
        """Test equality comparison with non-Calculation objects."""
        calc = Calculation('Addition', Decimal('5'), Decimal('3'))
        
        # Test with different types
        assert calc != "Addition(5, 3) = 8"
        assert calc != 123
        assert calc != None
        assert calc != {"operation": "Addition", "operand1": "5", "operand2": "3"}
    
    def test_eq_not_implemented_for_different_types(self):
        """Test that __eq__ returns NotImplemented for non-Calculation objects."""
        calc = Calculation('Addition', Decimal('5'), Decimal('3'))
        
        # The != operator should work fine, but == should return False or NotImplemented
        result = calc.__eq__("not a calculation")
        assert result is NotImplemented
        
        result = calc.__eq__(123)
        assert result is NotImplemented
        
        result = calc.__eq__(None)
        assert result is NotImplemented
    
    def test_eq_reflexivity(self):
        """Test that a calculation equals itself."""
        calc = Calculation('Addition', Decimal('5'), Decimal('3'))
        assert calc == calc
    
    def test_eq_symmetry(self):
        """Test that equality is symmetric."""
        calc1 = Calculation('Addition', Decimal('5'), Decimal('3'))
        calc2 = Calculation('Addition', Decimal('5'), Decimal('3'))
        
        assert calc1 == calc2
        assert calc2 == calc1
    
    def test_eq_transitivity(self):
        """Test that equality is transitive."""
        calc1 = Calculation('Addition', Decimal('5'), Decimal('3'))
        calc2 = Calculation('Addition', Decimal('5'), Decimal('3'))
        calc3 = Calculation('Addition', Decimal('5'), Decimal('3'))
        
        assert calc1 == calc2
        assert calc2 == calc3
        assert calc1 == calc3
    
    def test_eq_complex_calculations(self):
        """Test equality with complex calculations."""
        calc1 = Calculation('Division', Decimal('10'), Decimal('3'))
        calc2 = Calculation('Division', Decimal('10'), Decimal('3'))
        
        assert calc1 == calc2
    
    def test_eq_with_negative_numbers(self):
        """Test equality with negative numbers."""
        calc1 = Calculation('Subtraction', Decimal('-5'), Decimal('-3'))
        calc2 = Calculation('Subtraction', Decimal('-5'), Decimal('-3'))
        
        assert calc1 == calc2
    
    def test_eq_with_zero_operands(self):
        """Test equality with zero operands."""
        calc1 = Calculation('Multiplication', Decimal('0'), Decimal('5'))
        calc2 = Calculation('Multiplication', Decimal('0'), Decimal('5'))
        
        assert calc1 == calc2