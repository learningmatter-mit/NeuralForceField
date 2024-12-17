
import os
import pytest
import torch

torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", 1)))


def pytest_addoption(parser):
    print("add option")
    parser.addoption("--device", action="store", default="cpu", help="Whether to use the CPU or GPU for the tests")


@pytest.fixture
def device(request):
    return request.config.getoption("--device")
