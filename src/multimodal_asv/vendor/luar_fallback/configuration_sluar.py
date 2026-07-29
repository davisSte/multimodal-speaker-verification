from transformers import PretrainedConfig, AutoConfig, AutoTokenizer, BertTokenizer, BertTokenizerFast
import torch

class SluarConfig(PretrainedConfig):
    model_type = "sluar"

    def __init__(
        self,
        model_name_or_path="bert-base-uncased",
        embedding_dim=512,
        epsilon=1e-9,
        torch_dtype="bfloat16",
        gradient_checkpointing=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_name_or_path = model_name_or_path
        self.embedding_dim = embedding_dim
        self.epsilon = epsilon
        self.torch_dtype = torch_dtype
        self.gradient_checkpointing = gradient_checkpointing

try:
    AutoConfig.register("sluar", SluarConfig)
except ValueError:
    pass

try:
    AutoTokenizer.register(SluarConfig, BertTokenizer, BertTokenizerFast)
except ValueError:
    pass
