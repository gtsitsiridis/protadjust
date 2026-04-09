import logging
import pytest


@pytest.fixture(autouse=True)
def setup_logger():
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logging.getLogger().setLevel(logging.DEBUG)
