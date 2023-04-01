import src.data_cleaner as dc
import pandas as pd
import numpy as np
import pytest

@pytest.fixture
def data():
    return pd.DataFrame(
        data = np.array([
            ['Square', 4, 'red', 10],
            ['Triangle', 3, 'blue', 5],
            ['Circle', None, 'orange', 1],
            ['Pentagon', 5, 'green', 20]
        ]), columns = ['Shape Name', 'Vertices', 'Color', 'Quantity']
    )

def test_remove_rows_with_null_values_standard(data):
    cleaned_data = dc.remove_rows_with_null_values(data)
    assert len(cleaned_data.index) == 3

