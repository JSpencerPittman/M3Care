from torchtyping import TensorType  # type: ignore

# TimeEmbTensor = TensorType["time_dim", "embed_dim"]
# BatTimeEmbTensor = TensorType["batch_size", "time_dim", "embed_dim"]
# BatTimePosencTensor = TensorType["batch_size", "time_dim", "pos_encode_dim"]
# BatTimeSenTensor = TensorType["batch_size", "time_dim", "sensor_dim"]
# EdgeTimeEmbTensor = TensorType["num_edges", "time_dim", "embed_dim"]
# TimeSenEmbTensor = TensorType["time_dim", "sensor_dim", "embed_dim"]

BatTimeTensor = TensorType["batch_size", "time_dim"]
EdgeTimeTensor = TensorType["num_edges", "time_dim"]
TimePeTensor = TensorType["time_dim", "pos_encode_dim"]

BatSenOutTensor = TensorType["batch_size", "sensor_dim", "out_dim"]
BatSenSenTensor = TensorType["batch_size", "sensor_dim", "sensor_dim"]
BatSenTimeTensor = TensorType["batch_size", "sensor_dim", "time_dim"]
BatTimePeTensor = TensorType["batch_size", "time_dim", "pos_encode_dim"]
BatTimeSenTensor = TensorType["batch_size", "time_dim", "sensor_dim"]
EdgeTimeObsTensor = TensorType["num_edges", "time_dim", "obs_dim"]
TimeSenObsTensor = TensorType["time_dim", "num_sensors", "obs_dim"]

BatHeadSenSenTensor = TensorType["batch_size", "num_heads", "sensor_dim", "sensor_dim"]
BatHeadSenTimeTensor = TensorType["batch_size", "num_heads", "sensor_dim", "time_dim"]
BatTimeSenObsTensor = TensorType["batch_size", "time_dim", "sensor_dim", "obs_dim"]
BatTimeSenObs_EmbTensor = TensorType["batch_size",
                                     "time_dim",
                                     "sensor_dim",
                                     "obs_embed_dim"]
BatTimeSenSenTensor = TensorType["batch_size", "time_dim", "sensor_dim", "sensor_dim"]

BatHeadSenTimeObs_Pe_EmbTensor = TensorType["batch_size",
                                            "head_dim",
                                            "sensor_dim",
                                            "time_dim",
                                            "obs_pe_dim"]
BatHeadTimeSenObs_EmbTensor = TensorType["batch_size",
                                         "num_heads",
                                         "time_dim",
                                         "sensor_dim",
                                         "obs_emb_dim"]
BatHeadTimeSenSenTensor = TensorType["batch_size",
                                     "num_heads",
                                     "time_dim",
                                     "sensor_dim",
                                     "sensor_dim"]
