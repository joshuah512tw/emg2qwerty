import torch
import pytest
from emg2qwerty.modules import LSTMCell, LSTMLayer, LSTMEncoder

@pytest.mark.parametrize(
    "N, input_size, hidden_size",
    [
        (1, 8, 16),
        (4, 32, 64),
        (8, 128, 256),
    ],
)
def test_lstm_cell_output_shapes(N: int, input_size: int, hidden_size: int):
    cell = LSTMCell(input_size, hidden_size)
    x = torch.randn(N, input_size)
    h, c = torch.zeros(N, hidden_size), torch.zeros(N, hidden_size)
    h_new, c_new = cell(x, h, c)
    assert h_new.shape == (N, hidden_size)
    assert c_new.shape == (N, hidden_size)


@pytest.mark.parametrize(
    "T, N, input_size, hidden_size",
    [
        (1, 1, 8, 16),
        (20, 4, 48, 64),
        (100, 8, 128, 256),
    ],
)
def test_lstm_layer_output_shape(T: int, N: int, input_size: int, hidden_size: int):
    layer = LSTMLayer(input_size, hidden_size)
    assert layer(torch.randn(T, N, input_size)).shape == (T, N, hidden_size)


@pytest.mark.parametrize(
    "T, N, input_size, hidden_size, num_layers",
    [
        (10, 1, 16, 32, 1),
        (30, 4, 48, 64, 2),
        (100, 8, 128, 256, 3),
    ],
)
def test_lstm_encoder_output_shape(
    T: int, N: int, input_size: int, hidden_size: int, num_layers: int
):
    encoder = LSTMEncoder(input_size, hidden_size, num_layers)
    assert encoder(torch.randn(T, N, input_size)).shape == (T, N, hidden_size)