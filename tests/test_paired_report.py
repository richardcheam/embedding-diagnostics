"""Report wiring uses stored records only; floor definitions preserve positives."""
import runpy

import numpy as np
import pytest


def test_gallery_positive_floors_and_query_change_counts():
    functions=runpy.run_path('scripts/report_image_text_replication.py')
    floors=functions['floors'](1000,5000,5)
    assert floors['t2i']['hit@10'] == .01
    assert floors['i2t']['set_recall@10'] == .002
    assert floors['i2t']['hit@1'] == pytest.approx(.001)
    assert .0099 < floors['i2t']['hit@10'] < .0101
    changes=functions['changes']([0,1,1,0],[1,0,1,0])
    assert changes == {'gain':1,'loss':1,'unchanged':2}
    assert np.isfinite(list(floors['i2t'].values())).all()
