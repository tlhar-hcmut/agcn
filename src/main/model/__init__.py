from .stream_spatial import *
from .stream_temporal import *


class TKNet(torch.nn.Module):
    def __init__(
        self,
        num_joint=25,
        num_channel=3,
        num_class=60,
        cls_graph=None,
        graph_args=dict(),
    ):
        super(TKNet, self).__init__()

        if cls_graph is None:
            raise ValueError()
        else:
            self.graph = cls_graph(**graph_args)

        # stream old
        self.stream_spatital = StreamSpatialGCN(
            num_class=num_class, cls_graph=cls_graph,
        )
        self.stream_temporal = StreamTemporalGCN(
            num_joint=num_joint, num_channel=num_channel
        )

        self.fc = nn.Linear(24, num_class)

    def forward(self, x):
        return self.fc(torch.cat((self.stream_spatital, self.stream_temporal), 1))
