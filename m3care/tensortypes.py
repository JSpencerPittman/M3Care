from torchtyping import TensorType

EmbeddedStaticTensor = TensorType["batch_size", "embed_dim"]
EmbeddedSequentialTensor = TensorType["batch_size", "time_dim", "embed_dim"]
EmbeddedTensor = EmbeddedStaticTensor | EmbeddedSequentialTensor

MaskStaticTensor = TensorType["batch_size"]
MaskSequentialTensor = TensorType["batch_size", "time_dim"]
MaskTensor = MaskStaticTensor | EmbeddedSequentialTensor

ModalsMaskTensor = TensorType["modal_num", "batch_size"]
BatchSimilarityTensor = TensorType["batch_size", "batch_size"]
BatchSimilarityMaskTensor = TensorType["batch_size", "batch_size"]

MultiModalTensor = TensorType["batch_size", "seq_dim", "embed_dim"]
MultiModalMaskTensor = TensorType["batch_size", "seq_dim", "seq_dim"]

Scalar = TensorType[()]
