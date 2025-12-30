
import torch
from metacontroller.metacontroller import Transformer, MetaController

def test_metacontroller():

    ids = torch.randint(0, 256, (1, 1024))

    model = Transformer(
        512,
        embed = dict(num_discrete = 256),
        lower_body = dict(depth = 2,),
        upper_body = dict(depth = 2,),
        readout = dict(num_discrete = 256)
    )

    meta_controller = MetaController(512)

    logits = model(ids, meta_controller = meta_controller)

    assert logits.shape == (1, 1024, 256)
