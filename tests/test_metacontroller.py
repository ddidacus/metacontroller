
import torch
from metacontroller.metacontroller import Transformer

def test_metacontroller():

    ids = torch.randint(0, 256, (1, 1024))

    model = Transformer(
        256,
        embed = dict(num_discrete = 256),
        lower_body = dict(depth = 2,),
        upper_body = dict(depth = 2,),
        readout = dict(num_discrete = 256)
    )

    logits = model(ids)
    assert logits.shape == (1, 1024, 256)
