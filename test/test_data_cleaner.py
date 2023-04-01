import src.data_cleaner as dc
import pandas as pd
import numpy as np
import pytest

@pytest.fixture
def standard_data():
    return pd.DataFrame(
        data = np.array([
            ['Square', 4, 'red', 10],
            ['Triangle', 3, 'blue', 5],
            ['Circle', None, 'orange', 1],
            ['Pentagon', 5, 'green', 20]
        ]), columns = ['Shape Name', 'Vertices', 'Color', 'Quantity']
    )

@pytest.fixture
def empty_data():
    return pd.DataFrame(
        data = np.array([])
    )

def test_remove_rows_with_null_values_standard(standard_data):
    cleaned_data = dc.remove_rows_with_null_values(standard_data)
    assert len(cleaned_data.index) == 3

def test_remove_rows_with_null_values_empty(empty_data):
    cleaned_data = dc.remove_rows_with_null_values(empty_data)
    assert len(cleaned_data.index) == 0
