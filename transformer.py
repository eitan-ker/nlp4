from torch import nn
import torch
import torch.nn.functional as F # Added for better_sample_continuation
import attention
import mlp

class TransformerDecoderBlock(nn.Module):
    def __init__(self, n_heads: int, embed_size: int, mlp_hidden_size: int, max_context_len, with_residuals: bool = False):
        super().__init__()
        self.causal_attention = attention.CausalSelfAttention(embed_size, n_heads, max_context_len)
        self.mlp = mlp.MLP(embed_size, mlp_hidden_size)
        self.layer_norm_1 = nn.LayerNorm(embed_size)
        self.layer_norm_2 = nn.LayerNorm(embed_size)
        self.with_residuals = with_residuals

    ### MODIFIED ###
    # Implemented the forward pass with residual connections.
    def forward(self, inputs):
        if self.with_residuals:
            # First residual connection (Pre-Norm): LayerNorm -> Attention -> Add
            x = inputs + self.causal_attention(self.layer_norm_1(inputs))
            # Second residual connection (Pre-Norm): LayerNorm -> MLP -> Add
            x = x + self.mlp(self.layer_norm_2(x))
            return x
        else:
            # This block is not used when with_residuals=True
            x = inputs
            x = self.causal_attention(self.layer_norm_1(x))
            x = self.mlp(self.layer_norm_2(x))
            return x

### MODIFIED ###
# Completed the entire Embed class.
class Embed(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, max_context_len):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embed_size)
        self.position_embeddings = nn.Embedding(max_context_len, embed_size)
        self.max_context_len = max_context_len

    def forward(self, x):
        # x has the shape (b x n) where b is batch dimension and n is sequence length.
        B, N = x.size()
        device = x.device

        # Create positional indices (0, 1, 2, ..., N-1)
        positions = torch.arange(0, N, dtype=torch.long, device=device)

        # Look up the embeddings for tokens and positions
        tok_embeddings = self.token_embeddings(x)
        pos_embeddings = self.position_embeddings(positions)

        # The final embedding is the sum of token and position embeddings
        return tok_embeddings + pos_embeddings


class TransformerLM(nn.Module):
    def __init__(
            self,
            n_layers: int,
            n_heads: int,
            embed_size: int,
            max_context_len: int,
            vocab_size: int,
            mlp_hidden_size: int,
            with_residuals: bool,
            ):
        super().__init__()
        self.embed = Embed(vocab_size, embed_size, max_context_len)
        self.layers = nn.ModuleList([TransformerDecoderBlock(n_heads, embed_size, mlp_hidden_size, max_context_len, with_residuals) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(embed_size)
        self.word_prediction = nn.Linear(embed_size, vocab_size)
        self.max_context_len = max_context_len

        self.init_weights()

        n_params = sum(p.numel() for p in self.parameters())
        print("Parameter count: %.2fM" % (n_params/1e6,))

    def forward(self, inputs: torch.IntTensor): # Type hint corrected to torch.IntTensor
        x = self.embed(inputs)
        for layer in self.layers:
            x = layer(x)
        x = self.layer_norm(x)
        logits = self.word_prediction(x)
        return logits

    ### MODIFIED ###
    # Corrected and simplified the weight initialization.
    def init_weights(self):
        for pn, p in self.named_parameters():
            if p.dim() > 1: # Initialize weights of layers with more than one dimension
                 torch.nn.init.xavier_normal_(p)

        # Initialize biases for Linear layers to zero
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                torch.nn.init.zeros_(m.bias)


    def sample_continuation(self, prefix: list[int], max_tokens_to_generate: int) -> list[int]:
        feed_to_lm = prefix[:]
        generated = []
        self.eval() # Set model to evaluation mode for sampling
        with torch.no_grad():
            while len(generated) < max_tokens_to_generate:
                if len(feed_to_lm) > self.max_context_len:
                    # if we have more tokens than context length, trim it to context length.
                    feed_to_lm = feed_to_lm[-self.max_context_len:]
                logits = self(torch.tensor([feed_to_lm], dtype=torch.long)) # dtype corrected to long
                logits_for_last_token = logits[0][-1]
                distribution_for_last_token = F.softmax(logits_for_last_token, dim=0) # dim specified
                sampled_token = torch.multinomial(distribution_for_last_token, num_samples=1)
                generated.append(sampled_token.item()) # .item() to get the integer value
                feed_to_lm.append(sampled_token.item())
        self.train() # Set model back to training mode
        return generated

    def better_sample_continuation(self, prefix: list[int], max_tokens_to_generate: int, temperature: float, topK: int) -> list[int]:
        raise Exception("Not implemented")
        # TODO implement this.
        # Temperature should be the temperature in which you sample.
        # TopK indicates that we don't sample from the entire distribution, but only from the top k scoring tokens
        # for the given position.
