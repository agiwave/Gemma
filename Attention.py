import aka.nn as nn
import aka.numpy as np

def AttentionBlock(args):
    '''
    Group-Query Attention
    Args:
        args.latent_dim 
        args.attn_hidden_dim(Optional, default: latent_dim)

    Examples:
        default ==> Attention
        args.attn_heads = 8 ==> MHA: Multi-Head Attention
        args.attn_kv_groups = 1 ==> MQA: Multi-Query Attention
        args.attn_kv_groups = 2 ==> GQA: Group-Query Attention
    '''
    def apply_rotary_emb(x, freqs_cis):
        '''
        Reference: LLaMA and Gemma
        Applies the rotary embedding to the query and key tensors.
        '''
        B,L,N,D = x.shape
        y = np.reshape(x, (B,L,N,2,D//2)).float()
        y = np.einsum('blncd->bnldc',y)
        y = np.view_as_complex(y.contiguous())
        y = np.view_as_real(y*freqs_cis).type_as(x)
        y = np.einsum('bnldc->blncd', y)
        return np.reshape(y, (B,L,N,D))

    def forward(self, x, *, freqs_cis=None, state=None):
        B, L, _ = x.size()

        # -- qkv --
        attn_hidden_dim, attn_heads, attn_kv_groups = self.attn_hidden_dim, self.attn_heads, self.attn_kv_groups
        attn_head_dim = attn_hidden_dim // attn_heads
        attn_kv_dim = attn_head_dim * attn_kv_groups
        q, k, v  = self.c_attn(x).split([attn_hidden_dim,attn_kv_dim,attn_kv_dim], dim=2)
        q = q.view(B, L, attn_heads, attn_head_dim)
        k = k.view(B, L, attn_kv_groups, attn_head_dim)
        v = v.view(B, L, attn_kv_groups, attn_head_dim)

        # -- append state cache --
        if state is not None:
            if 'kv_cache' in state:
                k_cache, v_cache = state['kv_cache']
                B, L, N, D = k.shape
                if B == k_cache.size(0):
                    k = np.cat((k_cache.detach()[:,L-self.block_size:,:], k), dim=1)
                    v = np.cat((v_cache.detach()[:,L-self.block_size:,:], v), dim=1)
            state['kv_cache'] = (k, v)

        # -- rotary embedding --
        M = k.size(1)
        if freqs_cis is not None:
            q = apply_rotary_emb(q, freqs_cis=freqs_cis[M-L:M,:])
            k = apply_rotary_emb(k, freqs_cis=freqs_cis[:M,:])

        # -- repeat kv to q --
        if attn_kv_groups != attn_heads:
            # [B, L, N, head_dim]
            k = np.repeat(k, attn_heads // attn_kv_groups, dim=2)
            v = np.repeat(v, attn_heads // attn_kv_groups, dim=2)

        # -- attn --
        att = np.einsum('blnd,bmnd->bnlm', q, k) * self.scale_dk
        att = att.masked_fill(self.bias[:,:,M-L:M,:M]==0, float('-inf'))
        att = np.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = np.einsum('bnlm,bmnd->blnd', att, v)
        y = y.reshape(B, L, attn_hidden_dim) # re-assemble all head outputs side by side
        return self.resid_dropout(self.c_proj(y))

    
    attn_args = args.attn_args
    attn_hidden_dim = getattr(attn_args, 'attn_hidden_dim', args.latent_dim)
    attn_heads = getattr(attn_args, 'attn_heads', 1)
    attn_kv_groups = getattr(attn_args, 'attn_kv_groups', attn_heads)
    bias = getattr(args, 'bias', False)
    dropout = getattr(args, 'dropout', 0.2)
    attn_head_dim = attn_hidden_dim//attn_heads
    assert attn_head_dim * attn_heads == attn_hidden_dim
    assert attn_heads % attn_kv_groups == 0
    return nn.Module( 
        forward = forward, 
        c_attn = nn.Linear(args.latent_dim, attn_head_dim * (attn_heads + attn_kv_groups * 2), bias=bias),
        c_proj = nn.Linear(attn_hidden_dim, args.latent_dim, bias=bias),
        attn_dropout = nn.Dropout(dropout),
        resid_dropout = nn.Dropout(dropout),
        attn_hidden_dim = attn_hidden_dim,
        attn_heads = attn_heads,
        attn_kv_groups = attn_kv_groups,
        block_size = args.block_size,
        scale_dk = 1.0/np.sqrt(np.array([attn_head_dim])),
        bias = np.tril(np.ones(1,1,args.block_size, args.block_size))
    )


# --- Example ---
if __name__ == "__main__":
    class Args():
        def __init__(self, **kwargs): 
            for key in kwargs: setattr(self, key, kwargs[key])
    atten = AttentionBlock(Args(
        latent_dim = 384,
        block_size = 256,
        attn_args = Args(
            attn_heads = 6,
            attn_kv_groups = 3,
            attn_hidden_dim = 384,
        ),
    ))
    input = np.randn(50, 100, 384)
    output = atten(input)
    print(output.size())