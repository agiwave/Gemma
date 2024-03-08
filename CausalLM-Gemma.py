import aka.nn as nn
import aka.numpy as np

def GemmaArgs(name):
    class Args():
        def __init__(self, **kwargs): 
            for key in kwargs: setattr(self, key, kwargs[key])

    from sentencepiece import SentencePieceProcessor
    args = Args(
        tokenizer = SentencePieceProcessor('data/tokenizer.model'),
        vocab_size = 256000,
        block_size = 8192,  # Limit Attention Seq Length to 256. Gemma2b --> 8192
        latent_dim = 2048,
        position_embedding = False,
        rotary_embedding = True,
        prev_norm = True,

        layers = ['Attention', 'MLP']*18,
        attn_heads = 8,  #
        attn_kv_groups = 1,
        attn_hidden_dim = 2048,
        attn_cache_kv = True,
        mlp_heads = 1,
        mlp_hidden_dim = 16384,
        mlp_gate = True,

        dropout = 0.2,
        bias = False, # bias in Linear?
        rope_theta = 10000,
    )
    match name:
        case '2b':
            args.layers = ['Attention', 'MLP']*18
            args.latent_dim = 2048
            args.attn_heads = 8
            args.attn_hidden_dim = 2048
            args.attn_kv_groups = 1
            args.mlp_hidden_dim = 16384
            return args
        case '8b':
            args.layers = ['Attention', 'MLP']*28
            args.latent_dim = 3072
            args.attn_heads = 16
            args.attn_hidden_dim = 3072
            args.attn_kv_groups = 16
            args.mlp_hidden_dim = 24576
            return args

        case _:
            assert False, f"Unknown Gemma name{name}"

def Gemma(name, ckpt=None):
    from CausalLM import CausalLM
    gemma = CausalLM(GemmaArgs(name))
    if ckpt is not None:
        def copy(desc, src):
            if not (desc.shape == src.shape):
                print(desc.shape, src.shape)
                assert False
            desc.copy_(src)

        state = np.load(
            ckpt, mmap=True, weights_only=True,
        )['model_state_dict']

        '''
        Notice: All RMSNorm weight in gemma shoud be added by 1 first. See reason below(in Gemma):
            def RMSNorm(dim: int, eps: float = 1e-6, add_unit_offset: bool = True):
                def forward(self, x):
                    x = (x.float() * np.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x)
                    if self.add_unit_offset:
                        output = x * (1 + self.weight) # Look here :)
                    else:
                        output = x * self.weight
                    return output
                return nn.Module(
                    forward = forward,
                    eps = eps,
                    add_unit_offset = add_unit_offset,
                    weight = nn.Parameter(np.ones(dim)))
        ''' 
        with np.no_grad():
            copy(gemma.embedding.weight, state['embedder.weight'])
            copy(gemma.post_norm.weight, state['model.norm.weight']+1)
            for i in range(len(gemma.layers)//2):
                copy(gemma.layers[i*2].normalize.weight, state[f'model.layers.{i}.input_layernorm.weight']+1)
                copy(gemma.layers[i*2].layer.c_attn.weight, state[f'model.layers.{i}.self_attn.qkv_proj.weight'])
                copy(gemma.layers[i*2].layer.c_proj.weight, state[f'model.layers.{i}.self_attn.o_proj.weight'])
                copy(gemma.layers[i*2+1].normalize.weight, state[f'model.layers.{i}.post_attention_layernorm.weight']+1)
                copy(gemma.layers[i*2+1].layer.gate_proj.weight, state[f'model.layers.{i}.mlp.gate_proj.weight'])
                copy(gemma.layers[i*2+1].layer.up_proj.weight, state[f'model.layers.{i}.mlp.up_proj.weight'])
                copy(gemma.layers[i*2+1].layer.down_proj.weight, state[f'model.layers.{i}.mlp.down_proj.weight'])
    return gemma

if __name__ == "__main__":
    with np.no_grad():
        gemma = Gemma('2b', 'data/gemma-2b-it.ckpt')
        print('Model loaded')
        print(gemma.generate("The self-attention is important for transformer because"))
