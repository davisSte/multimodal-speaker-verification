import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from transformers import AutoModel as HFAutoModel
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
import math
from typing import Tuple, Optional
from .configuration_sluar import SluarConfig
from transformers.models.auto.modeling_auto import AutoModel as TransformersAutoModel


class SentenceTransformerWrapper(nn.Module):
    """
    A wrapper for sentence-transformer models to make them compatible with the
    evaluation script.
    """

    def __init__(self, model_name_or_path: str, **kwargs):
        super().__init__()
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name_or_path, trust_remote_code=True)

    def forward(
        self, text: list[list[str]], **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Embeds episodes of sentences and returns the averaged embedding.

        Args:
            text: A list of episodes, where each episode is a list of sentences.
                  Example: [ ['sent1', 'sent2'], ['sent3', 'sent4'] ]

        Returns:
            A tuple containing:
            - The episode-level embeddings (averaged sentence embeddings).
            - None (to match the expected output format).
        """
        episode_embeddings = []
        for episode in text:
            # The model.encode function returns a numpy array, convert it to a tensor
            embeddings = self.model.encode(
                episode, convert_to_tensor=True, show_progress_bar=False
            )
            # Average the sentence embeddings to get the episode embedding
            episode_embedding = embeddings.mean(dim=0)
            episode_embeddings.append(episode_embedding)

        # Stack the episode embeddings into a single tensor
        final_embeddings = torch.stack(episode_embeddings)
        return (final_embeddings, None)

    def similarity(self, queries: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes cosine similarity between two sets of embeddings."""
        queries_norm = F.normalize(queries, p=2, dim=1)
        targets_norm = F.normalize(targets, p=2, dim=1)
        return torch.matmul(queries_norm, targets_norm.transpose(0, 1))


class PretrainedLUARWrapper(nn.Module):
    """
    A wrapper for pretrained LUAR models from HuggingFace to make them
    compatible with the evaluation script.
    """

    def __init__(self, model_name_or_path: str, **kwargs):
        super().__init__()
        self.model = HFAutoModel.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Embeds episodes of sentences and returns the averaged embedding.

        Args:
            input_ids: A tensor of shape (B, N, E, L)
            attention_mask: A tensor of shape (B, N, E, L)

        Returns:
            A tuple containing:
            - The episode-level embeddings.
            - None (to match the expected output format).
        """
        # The pretrained LUAR model expects input of shape (B, E, L).
        # During evaluation, N (num_views) is 1, so we squeeze it.
        if input_ids.dim() == 4 and input_ids.shape[1] == 1:
            input_ids = input_ids.squeeze(1)
            attention_mask = attention_mask.squeeze(1)

        embeddings = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # The HF model might return a tuple. The first element should be the embeddings.
        if isinstance(embeddings, (list, tuple)):
            embeddings = embeddings[0]
        return (embeddings, None)

    def similarity(self, queries: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes cosine similarity between two sets of embeddings."""
        queries_norm = F.normalize(queries, p=2, dim=1)
        targets_norm = F.normalize(targets, p=2, dim=1)
        return torch.matmul(queries_norm, targets_norm.transpose(0, 1))


class AutoModel(nn.Module):

    @staticmethod
    def from_pretrained(model_name_or_path: str, **kwargs):
        return HFAutoModel.from_pretrained(model_name_or_path, **kwargs)


class SelfAttention(nn.Module):
    """
    Implements a basic self-attention mechanism.

    The attention is computed as: softmax(K @ Q.T / sqrt(d_k)) @ V.
    Note the order of K and Q in the first matrix multiplication, which defines
    the output shape.
    """

    def __init__(self, dtype=None):
        super(SelfAttention, self).__init__()
        self.dtype = dtype

    def forward(
        self, k: torch.Tensor, q: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """
        Performs the self-attention operation using the formula:
        output = softmax(K @ Q.T / sqrt(d_k)) @ V

        Args:
            k: Key tensor (K). Expected shape: (batch, seq_len_k, feature_dim).
            q: Query tensor (Q). Expected shape: (batch, seq_len_q, feature_dim).
               The feature_dim of Q must match feature_dim of K.
            v: Value tensor (V). Expected shape: (batch, seq_len_v, value_dim).
               For the final matrix multiplication (p_attn @ V), seq_len_q must
               be equal to seq_len_v.

        Returns:
            The output tensor after attention.
            Shape: (batch, seq_len_k, value_dim).
        """
        if self.dtype:
            k = k.to(self.dtype)
            q = q.to(self.dtype)
            v = v.to(self.dtype)
        d_k = q.size(-1)  # feature_dim of query (and key)
        # scores = K @ Q.T
        # K: (B, S_k, D_k), Q: (B, S_q, D_k) -> Q.T: (B, D_k, S_q)
        # scores: (B, S_k, S_q)
        scores = torch.matmul(k, q.transpose(-2, -1)) / math.sqrt(d_k)
        # p_attn: (B, S_k, S_q), softmax over S_q dimension
        p_attn = F.softmax(scores, dim=-1)
        # output = p_attn @ V
        # p_attn: (B, S_k, S_q), V: (B, S_v, D_v)
        # Requires S_q == S_v for matmul.
        # Output: (B, S_k, D_v)
        return torch.matmul(p_attn, v)


class LUAR(nn.Module, PyTorchModelHubMixin):
    """
    This model processes episodes (views) of documents from an author to create
    a fixed-size embedding for each episode. It uses a pre-trained transformer
    model to get document embeddings, applies mean pooling to these token
    embeddings, then uses a self-attention mechanism over the document
    embeddings within an episode, followed by max pooling and a final linear layer
    to produce the episode embedding.

    """

    def __init__(
        self,
        model_name_or_path: str,
        embedding_dim: int,
        epsilon: float = 1e-9,
        torch_dtype: str = "bfloat16",
        gradient_checkpointing: bool = False,
        **kwargs,
    ):
        """
        Initializes the LUAR model.

        Args:
            model_name_or_path: Path or identifier for the pre-trained HuggingFace
                                transformer model (e.g., "bert-base-uncased").
            embedding_dim: The dimension of the final output embedding for each episode.
            epsilon: A small value for numerical stability in mean_pooling.
        """
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.epsilon = epsilon

        # Correctly handle torch_dtype
        if isinstance(torch_dtype, str):
            self.torch_dtype = getattr(torch, torch_dtype, None)
        else:
            self.torch_dtype = torch_dtype

        self.gradient_checkpointing = gradient_checkpointing
        self.create_transformer()
        self.attn_fn = SelfAttention(dtype=self.torch_dtype)
        self.linear = nn.Linear(self.hidden_size, embedding_dim, dtype=self.torch_dtype)

    def create_transformer(self):
        """
        Initializes the transformer model from the given `model_name_or_path`
        and sets `self.hidden_size` based on the transformer's configuration.
        """
        self.transformer = HFAutoModel.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
            torch_dtype=self.torch_dtype,
        )
        if self.gradient_checkpointing:
            self.transformer.gradient_checkpointing_enable()
        self.hidden_size = self.transformer.config.hidden_size

    def mean_pooling(
        self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Performs mean pooling of token embeddings using an attention mask.

        Args:
            token_embeddings: Tensor of token embeddings.
                              Shape: (batch_size, sequence_length, hidden_size).
            attention_mask: Tensor representing the attention mask.
                            Shape: (batch_size, sequence_length).

        Returns:
            Pooled embeddings tensor. Shape: (batch_size, hidden_size).
        """
        # input_mask_expanded: (batch_size, sequence_length, hidden_size)
        input_mask_expanded = repeat(
            attention_mask, "b l -> b l d", d=self.hidden_size
        ).float()
        # sum_embeddings: (batch_size, hidden_size)
        sum_embeddings = reduce(
            token_embeddings * input_mask_expanded, "b l d -> b d", "sum"
        )
        # sum_mask: (batch_size, hidden_size)
        sum_mask = torch.clamp(
            reduce(input_mask_expanded, "b l d -> b d", "sum"), min=self.epsilon
        )
        return (sum_embeddings / sum_mask).to(dtype=token_embeddings.dtype)

    def get_sequence_embeddings(self, input_ids=None, attention_mask=None):
        """
        Computes embeddings for a standard batch of sequences (2D input).

        Args:
            'input_ids': Tensor of shape (B, L)
                         B = batch_size
                         L = sequence_length
            'attention_mask': Tensor of shape (B, L)

        Returns:
            A tuple (sequence_embeddings, None):
            - sequence_embeddings: Tensor of shape (B, embedding_dim).
                                   This is the final embedding for each sequence.
            - None: To maintain a consistent signature with get_episode_embeddings.
        """
        # Get token embeddings from the transformer
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
        )
        token_embeddings = outputs["last_hidden_state"]

        # Mean pooling over token embeddings
        # (B, L, hidden_size) -> (B, hidden_size)
        doc_embeddings = self.mean_pooling(token_embeddings, attention_mask)

        # Final linear projection
        # (B, hidden_size) -> (B, embedding_dim)
        final_sequence_embeddings = self.linear(doc_embeddings)

        return (final_sequence_embeddings, None)

    def get_episode_embeddings(self, input_ids=None, attention_mask=None):
        """
        Processes a batch of data to compute episode and comment embeddings.

        Args:
            'input_ids' and 'attention_mask'.
                'input_ids': Tensor of shape (B, N, E, L)
                             B = batch_size (number of authors)
                             N = num_views per author
                             E = episode_length (documents per view)
                             L = sequence_length (max document tokens)
                'attention_mask': Tensor of shape (B, N, E, L)

        Returns:
            A tuple (episode_embeddings, comment_embeddings):
            - episode_embeddings: Tensor of shape (B*N, embedding_dim).
                                  This is the final embedding for each view/episode.
            - comment_embeddings: Tensor of shape (B*N, E, hidden_size).
                                  Embeddings for each document within each
                                  view/episode, after transformer processing
                                  and before episode-level attention.

        Shape Transformation Walkthrough:
        1. `input_ids`, `attention_mask` initial: (B, N, E, L)
        2. `input_ids`, `attention_mask` after `rearrange`: ((B*N*E), L)
           (Flattened for transformer input)
        3. `outputs['last_hidden_state']` from transformer: ((B*N*E), L, hidden_size)
           (Token embeddings for each document)
        4. `comment_embeddings` after `mean_pooling`: ((B*N*E), hidden_size)
           (One embedding per document)
        5. `comment_embeddings` after `rearrange` to group by episode:
           (B*N, E, hidden_size)
           (E document embeddings per episode/view)
        6. `episode_embeddings` after `self.attn_fn` (SelfAttention over E docs):
           (B*N, E, hidden_size) (SelfAttention's output shape depends on K, which is (B*N, E, H))
        7. `episode_embeddings` after `reduce(..., 'max')` (max pooling over E docs):
           (B*N, hidden_size)
           (One embedding per episode/view, before final projection)
        8. `episode_embeddings` after `self.linear`: (B*N, embedding_dim)
           (Final projected embedding for each episode/view)
        """
        if input_ids.dim() == 3:
            # This case occurs during validation/testing where the 'N' dimension
            # (number of views) is squeezed. We add it back in.
            input_ids = input_ids.unsqueeze(1)
            if attention_mask is not None:
                attention_mask = attention_mask.unsqueeze(1)

        B, N, E, _ = input_ids.shape  # B=batch, N=num_views, E=episode_length

        # Reshape for transformer: (B, N, E, L) -> (B*N*E, L)
        current_input_ids = rearrange(input_ids, "b n e l -> (b n e) l")
        current_attention_mask = rearrange(attention_mask, "b n e l -> (b n e) l")

        outputs = self.transformer(
            input_ids=current_input_ids,
            attention_mask=current_attention_mask,
            return_dict=True,
            output_hidden_states=True,  # Ensure last_hidden_state is available
        )

        # Mean pooling over token embeddings for each document:
        # ((B*N*E), L, hidden_size) -> ((B*N*E), hidden_size)
        doc_embeddings = self.mean_pooling(
            outputs["last_hidden_state"], current_attention_mask
        )

        # Group document embeddings by episode/view:
        # ((B*N*E), hidden_size) -> (B*N, E, hidden_size)
        # This is the `comment_embeddings` output.
        grouped_comment_embeddings = rearrange(
            doc_embeddings, "(b n e) d -> (b n) e d", b=B, n=N, e=E
        )

        # Self-attention over document embeddings within each episode:
        # Input k,q,v are all `grouped_comment_embeddings`
        # (B*N, E, hidden_size) -> (B*N, E, hidden_size)
        # (SelfAttention output shape is (batch, seq_len_k, value_dim), here seq_len_k is E)
        episode_level_attn_output = self.attn_fn(
            grouped_comment_embeddings,
            grouped_comment_embeddings,
            grouped_comment_embeddings,
        )

        # Max pooling over attended document embeddings in the episode:
        # (B*N, E, hidden_size) -> (B*N, hidden_size)
        episode_summary_embedding = reduce(
            episode_level_attn_output, "bne el d -> bne d", "max"
        )

        # Final linear projection:
        # (B*N, hidden_size) -> (B*N, embedding_dim)
        final_episode_embeddings = self.linear(episode_summary_embedding)

        return final_episode_embeddings, grouped_comment_embeddings

    def similarity(self, queries: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes cosine similarity between two sets of embeddings."""
        queries_norm = F.normalize(queries, p=2, dim=1)
        targets_norm = F.normalize(targets, p=2, dim=1)
        return torch.matmul(queries_norm, targets_norm.transpose(0, 1))

    def forward(self, input_ids=None, attention_mask=None):
        """
        Main forward pass for the LUAR model.

        This method dispatches to the appropriate embedding function based on the
        dimensionality of the input tensor.
        - 2D input: (batch, seq_len) -> get_sequence_embeddings
        - 3D/4D input: (batch, num_views, episode_len, seq_len) -> get_episode_embeddings

        Args:
            input_ids: The input token IDs.
            attention_mask: The attention mask.

        Returns:
            A tuple containing the primary embedding and secondary embeddings (or None).
        """
        if input_ids.dim() == 2:
            return self.get_sequence_embeddings(
                input_ids=input_ids, attention_mask=attention_mask
            )
        else:
            return self.get_episode_embeddings(
                input_ids=input_ids, attention_mask=attention_mask
            )


try:
    TransformersAutoModel.register(SluarConfig, LUAR)
except ValueError:
    pass


class LUARPacked(LUAR):
    """
    This model variant handles packed sequences by processing the entire
    concatenated sequence in a single pass. It uses a custom block-diagonal
    attention mask to prevent tokens from different documents from attending
    to each other. After getting contextualized token embeddings, it segments
    the sequence, applies mean pooling to each segment to get document
    embeddings, and then aggregates them using the standard LUAR attention-based
    pooling.
    """

    def _create_attention_mask(self, input_ids, padding_mask):
        """
        Creates a combined attention mask that incorporates both padding and
        document boundaries for packed sequences.
        """
        # Create a block-diagonal mask to separate documents.
        is_eos = input_ids == self.transformer.config.eos_token_id
        # Shift to find the start of the next document.
        is_doc_start_shifted = torch.zeros_like(is_eos)
        is_doc_start_shifted[:, 1:] = is_eos[:, :-1]
        is_doc_start = is_doc_start_shifted
        is_doc_start[:, 0] = True
        # doc_ids will be 0 for the first doc, 1 for the second, etc.
        doc_ids = torch.cumsum(is_doc_start.long(), dim=1) - 1

        # block_mask is 1 only if two tokens are in the same document.
        block_mask = (doc_ids.unsqueeze(2) == doc_ids.unsqueeze(1)).long()

        # Combine with the padding mask.
        expanded_padding_mask = padding_mask.unsqueeze(2) * padding_mask.unsqueeze(1)
        combined_mask = block_mask * expanded_padding_mask

        # The shape must be (B, 1, L, L) for HuggingFace transformers.
        return combined_mask.unsqueeze(1)

    def _create_position_ids(self, input_ids):
        """
        Creates position_ids for a packed sequence, resetting positions to 0
        at the start of each new document.
        """
        seq_len = input_ids.shape[1]
        is_eos = input_ids == self.transformer.config.eos_token_id
        # Shift to find the start of the next document.
        is_doc_start_shifted = torch.zeros_like(is_eos)
        is_doc_start_shifted[:, 1:] = is_eos[:, :-1]
        is_doc_start = is_doc_start_shifted
        is_doc_start[:, 0] = True

        # doc_ids will be 0 for the first doc, 1 for the second, etc.
        doc_ids = torch.cumsum(is_doc_start.long(), dim=1) - 1
        # Get the index of the start of each document.
        doc_start_indices = torch.where(is_doc_start)[1]
        # Look up the start index for each token's document.
        token_doc_start_indices = doc_start_indices[doc_ids]
        # Global positions range from 0 to seq_len-1.
        global_positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        # Position within each document is the global position minus the doc's start position.
        position_ids = global_positions - token_doc_start_indices
        return position_ids.long()

    def forward(self, input_ids=None, attention_mask=None):
        """
        Dispatches to the appropriate embedding function based on input shape.
        - 3D/4D inputs are treated as standard episode batches (for validation).
        - 2D inputs are treated as packed sequences (for training).
        """
        if input_ids.dim() > 2:
            return super().get_episode_embeddings(
                input_ids=input_ids, attention_mask=attention_mask
            )

        # --- Packed Sequence Logic (assumes 2D input) ---
        custom_attention_mask = self._create_attention_mask(input_ids, attention_mask)
        position_ids = self._create_position_ids(input_ids)

        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=custom_attention_mask,
            position_ids=position_ids,
            return_dict=True,
        )
        token_embeddings = outputs.last_hidden_state[0]  # Shape: (L_packed, H)

        # Segment, Pool, and Aggregate
        boundary_indices = (
            input_ids[0] == self.transformer.config.eos_token_id
        ).nonzero(as_tuple=True)[0]

        doc_embeddings = []
        start_idx = 0
        for end_idx in boundary_indices:
            if start_idx < end_idx:
                doc_token_embeds = token_embeddings[start_idx:end_idx, :].unsqueeze(0)
                doc_attn_mask = attention_mask[0, start_idx:end_idx].unsqueeze(0)
                pooled = self.mean_pooling(doc_token_embeds, doc_attn_mask)
                doc_embeddings.append(pooled.squeeze(0))
            start_idx = end_idx + 1

        # Handle the last document in the packed sequence
        if start_idx < input_ids.shape[1] and attention_mask[0, start_idx:].any():
            doc_token_embeds = token_embeddings[start_idx:, :].unsqueeze(0)
            doc_attn_mask = attention_mask[0, start_idx:].unsqueeze(0)
            pooled = self.mean_pooling(doc_token_embeds, doc_attn_mask)
            doc_embeddings.append(pooled.squeeze(0))

        if not doc_embeddings:
            return (
                torch.zeros(1, self.linear.out_features, device=input_ids.device),
                None,
            )

        grouped_comments = torch.stack(doc_embeddings).unsqueeze(0)
        attn_output = self.attn_fn(grouped_comments, grouped_comments, grouped_comments)
        summary_embedding = reduce(attn_output, "b e d -> b d", "max")
        final_embedding = self.linear(summary_embedding)

        return final_embedding, grouped_comments


class LUARPackedFlat(LUAR):
    """
    This model variant handles packed sequences by processing the entire
    concatenated sequence as a single unit. It applies mean pooling over all
    tokens in the packed sequence to produce a single, "flat" embedding. This
    is in contrast to the nested approach of `LUARPacked`, which processes
    each sub-sequence individually.
    """

    def forward(self, input_ids=None, attention_mask=None):
        """
        Processes a batch of packed sequences by applying mean pooling over each
        sequence to produce a single "flat" embedding per sequence.

        This model expects a 4D input tensor of shape (B, N, E, L), where E
        (episode_length) must be 1. The input is reshaped to (B*N, L)
        for processing.

        Args:
            input_ids: Tensor of shape (B, N, 1, L).
            attention_mask: The attention mask for the packed tensor.

        Returns:
            A tuple containing the final flat embedding and None.
        """
        # Reshape the input from (B, N, E, L) to (B*N*E, L) for processing.
        # This aligns the model with the unified data pipeline.
        if input_ids.dim() > 2:
            input_ids = rearrange(input_ids, "b n e l -> (b n e) l")
            if attention_mask is not None:
                attention_mask = rearrange(attention_mask, "b n e l -> (b n e) l")
        # Get token embeddings from the transformer for the entire packed sequence
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
        )
        token_embeddings = outputs["last_hidden_state"]

        # Mean pooling over all token embeddings in the packed sequence
        sequence_embedding = self.mean_pooling(token_embeddings, attention_mask)

        # Final linear projection
        final_embedding = self.linear(sequence_embedding)

        return (final_embedding, None)
