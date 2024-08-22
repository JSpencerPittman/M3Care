from torchtyping import TensorType

BatTensor = TensorType["batch_size"]

BatEmbTensor = TensorType["batch_size", "embed_dim"]

BatSeqTensor = TensorType["batch_size", "seq_dim"]
BatSeqEmbTensor = TensorType["batch_size", "seq_dim", "embed_dim"]
BatSeqFeatTensor = TensorType["batch_size", "seq_dim", "feat_dim"]

BatTimeTensor = TensorType["batch_size", "time_dim"]
BatTimeEmbTensor = TensorType["batch_size", "time_dim", "embed_dim"]
BatTimeSeqTensor = TensorType["batch_size", "time_dim", "seq_dim"]

BatBatTensor = TensorType["batch_size", "batch_size"]

Scalar = TensorType[()]
