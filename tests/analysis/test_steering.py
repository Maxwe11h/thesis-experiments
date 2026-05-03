"""Unit tests for analysis.phase4.steering."""
import numpy as np
import pandas as pd
import pytest

from analysis.phase4.steering import steering_rate


def _frame(values, condition, feature='intensification_ratio'):
    return pd.DataFrame({
        'condition': condition,
        f'bm_{feature}': values,
    })


def test_steers_toward_higher_reference():
    """When advised direction is 'up' and condition pushes feature higher than
    vanilla, steering rate should exceed 50%."""
    vanilla = _frame(np.full(100, 0.5), 'vanilla')
    cond = _frame(np.full(100, 0.7), 'neutral')
    df = pd.concat([vanilla, cond])
    rate = steering_rate(df, feature='intensification_ratio',
                         condition='neutral', vanilla='vanilla',
                         direction='up')
    assert rate == pytest.approx(100.0)


def test_steers_against_higher_reference():
    """If the condition's median is below vanilla's, steering rate should be 0."""
    vanilla = _frame(np.full(100, 0.7), 'vanilla')
    cond = _frame(np.full(100, 0.5), 'neutral')
    df = pd.concat([vanilla, cond])
    rate = steering_rate(df, feature='intensification_ratio',
                         condition='neutral', vanilla='vanilla',
                         direction='up')
    assert rate == pytest.approx(0.0)


def test_handles_lower_is_better():
    vanilla = _frame(np.full(100, 0.7), 'vanilla')
    cond = _frame(np.full(100, 0.3), 'neutral')
    df = pd.concat([vanilla, cond])
    rate = steering_rate(df, feature='intensification_ratio',
                         condition='neutral', vanilla='vanilla',
                         direction='down')
    assert rate == pytest.approx(100.0)


def test_drops_nans():
    vanilla = _frame(np.full(100, 0.5), 'vanilla')
    cond = _frame([0.7, np.nan, 0.7] * 33 + [0.7], 'neutral')
    df = pd.concat([vanilla, cond])
    rate = steering_rate(df, feature='intensification_ratio',
                         condition='neutral', vanilla='vanilla',
                         direction='up')
    # All non-NaN values are above vanilla's 0.5 → 100 %
    assert rate == pytest.approx(100.0)


def test_invalid_direction_raises():
    df = _frame([0.5, 0.6], 'neutral')
    with pytest.raises(ValueError):
        steering_rate(df, feature='intensification_ratio',
                      condition='neutral', vanilla='vanilla',
                      direction='sideways')
