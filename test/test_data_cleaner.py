import src.pre_processing as dc
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

def pre_p_forex(standard_data):
    cleaned_data = dc.pre_p_forex(standard_data)
    assert len(cleaned_data.index) == 3
