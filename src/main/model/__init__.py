from .stream_spatial import *
from .stream_temporal import *


class TKNet(torch.nn.Module):
    def __init__(
        self,
        device,
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
        self.stream_spatial = StreamSpatialGCN(num_class=num_class, cls_graph=cls_graph)
        
        self.stream_temporal = StreamTemporalGCN(device, cls_graph=cls_graph)

        output_layer_spatial = list(self.stream_spatial.children())[-1]
        output_layer_temporal = list(self.stream_temporal.children())[-1]

        num_unit_spatial = output_layer_spatial.weight.shape[-1]
        num_unit_temporal = output_layer_temporal.weight.shape[-1]
        
        self.fc = nn.Linear(num_unit_spatial+num_unit_temporal, num_class)

    def forward(self, x):
        output_stream_spatial = self.stream_spatial(x)
        output_stream_temporal = self.stream_temporal(x)
        output_concat = torch.cat((output_stream_spatial, output_stream_temporal), dim=1)
        return self.fc(output_concat)
