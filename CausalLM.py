import aka.nn as nn
import aka.numpy as np

def RMSNorm(dim: int, eps: float = 1e-6):
    '''
    Reference: LLaMA and Gemma
    '''
    def forward(self, x):
        x = (x.float() * np.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x)
        return x * self.weight
    return nn.Module(
        forward = forward,
        eps = eps,
        weight = nn.Parameter(np.ones(dim)))

def Attention(args):
    '''
    Group-Query Attention

    Args:
        args.latent_dim 
        args.attn_hidden_dim(Optional, default: latent_dim)
        args.attn_cache_kv True/False

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

    def forward(self, x, *, freqs_cis=None):
        B, L, _ = x.size()

        # -- qkv --
        attn_hidden_dim, attn_heads, attn_kv_groups = self.attn_hidden_dim, self.attn_heads, self.attn_kv_groups
        attn_head_dim = attn_hidden_dim // attn_heads
        attn_kv_dim = attn_head_dim * attn_kv_groups
        q, k, v  = self.c_attn(x).split([attn_hidden_dim,attn_kv_dim,attn_kv_dim], dim=2)
        q = q.view(B, L, attn_heads, attn_head_dim)
        k = k.view(B, L, attn_kv_groups, attn_head_dim)
        v = v.view(B, L, attn_kv_groups, attn_head_dim)

        # -- append cache(history) --
        if(self.attn_cache_kv):
            k, v = self.cache_kv(k, v)

        # -- rotary embedding --
        if freqs_cis is not None:
            M = k.size(1)
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

    def cache_kv(self, k, v):
        if not hasattr(self,'kv_cache'):
            self.kv_cache = (k, v)
        else:
            k_cache, v_cache = self.kv_cache
            B, L, N, D = k.shape
            if B == k_cache.size(0):
                k = np.cat((k_cache.detach()[:,L-self.block_size:,:], k), dim=1)
                v = np.cat((v_cache.detach()[:,L-self.block_size:,:], v), dim=1)
            self.kv_cache = (k, v)
        return k,v
    
    attn_hidden_dim = getattr(args, 'attn_hidden_dim', args.latent_dim)
    attn_heads = getattr(args, 'attn_heads', 1)
    attn_kv_groups = getattr(args, 'attn_kv_groups', attn_heads)
    bias = getattr(args, 'bias', False)
    dropout = getattr(args, 'dropout', 0.2)
    attn_head_dim = attn_hidden_dim//attn_heads
    assert attn_head_dim * attn_heads == attn_hidden_dim
    assert attn_heads % attn_kv_groups == 0
    return nn.Module( 
        forward = forward, 
        cache_kv = cache_kv,
        c_attn = nn.Linear(args.latent_dim, attn_head_dim * (attn_heads + attn_kv_groups * 2), bias=bias),
        c_proj = nn.Linear(attn_hidden_dim, args.latent_dim, bias=bias),
        attn_dropout = nn.Dropout(dropout),
        resid_dropout = nn.Dropout(dropout),
        attn_hidden_dim = attn_hidden_dim,
        attn_heads = attn_heads,
        attn_kv_groups = attn_kv_groups,
        attn_cache_kv = getattr(args,'attn_cache_kv', False),
        block_size = args.block_size,
        scale_dk = 1.0/np.sqrt(np.array([attn_head_dim])),
        bias = np.tril(np.ones(1,1,args.block_size, args.block_size))
    )

def MLP(args):
    '''
    Reference: Gemma, LLaMA
    Examples:
        args.mlp_gate == True ==> GateMLP
    '''
    def forward(self, x, **kwargs):
        up = self.up_proj(x)
        if(self.gate_proj is not None):
            gate = self.gate_proj(x)
            gate = np.gelu(gate)    # silu LLaMA ?
            up = gate * up
        else:
            up = np.gelu(up)
        return self.down_proj(up)

    mlp_hidden_dim = getattr(args, 'mlp_hidden_dim', args.latent_dim)
    gate = getattr(args, 'mlp_gate', False)
    bias = getattr(args,'bias', False)
    return nn.Module(
        forward = forward,
        gate_proj = None if not gate else nn.Linear(args.latent_dim, mlp_hidden_dim, bias=bias),
        up_proj = nn.Linear(args.latent_dim, mlp_hidden_dim, bias=bias),
        down_proj = nn.Linear(mlp_hidden_dim, args.latent_dim, bias=bias))

def MHMLP(args):
    '''
    Multi-Head MLP
    '''
    def forward(self, inputs, **kwargs):
        B, L, D = inputs.shape
        x = np.einsum('bld,dnh->blnh', x, self.q_proj)
        x = np.gelu(x)
        x = np.einsum('blnd,dnh->blnh', x, self.up_proj)
        x = np.gelu(x)
        x = np.einsum('blnh,hnd->blnd', x, self.down_proj)
        return x.reshape(inputs.shape)

    head_latent_dim = args.latent_dim // args.mlp_heads
    assert head_latent_dim * args.mlp_heads == args.latent_dim
    return nn.Module(
        forward = forward,
        q_proj = nn.Parameter(shape=(args.latent_dim, args.mlp_heads, head_latent_dim), requires_grad=True, initializer="xavier_uniform"),
        up_proj = nn.Parameter(shape=(head_latent_dim, args.mlp_heads, args.mlp_hidden_dim), requires_grad=True, initializer="xavier_uniform"),
        down_proj = nn.Parameter(shape=(args.mlp_hidden_dim, args.mlp_heads, head_latent_dim), requires_grad=True, initializer="xavier_uniform"))

def MetaLayer(name, args):
    '''
    Build resident meta layer by name. Include: GQA(Group-Query Attention), MLP, GateMLP, ...
    '''
    def forward(self, x, **kwargs):
        x = self.normalize(x)
        return self.layer(x, **kwargs)

    match name:
        case 'Attention':
            m = Attention(args)
        case 'MLP':
            m = MLP(args)
        case 'MHMLP':
            m = MHMLP(args)
        case _:
            assert False, f"Unknown layer:{name}"

    return nn.Module(
        forward = forward,
        normalize = RMSNorm(args.latent_dim),
        layer = m
    )

def CausalLM(args):
    '''
    Causal Language Model.
    '''
    def forward(self, inputs, targets=None):
        _, L = inputs.shape
        assert L-1 <= self.block_size, f"Input size:{L} too large. Max size: {self.block_size-1}"

        x = inputs

        # -- Shift inputs and targets --
        if(targets is not None):
            t = x[:,1:]
            x = x[:,:L-1]

        # -- Embedding and layers
        x = self.embedding(x)
        if self.prev_norm:
            x = x * (x.size(-1)**0.5)   # -- Gemma, Why? --
        if self.pe is not None:
            x = x + pe
        for l in self.layers:
            x = x + l(x, freqs_cis=self.freqs_cis)
        x = self.post_norm(x)

        # -- outputs and loss, why not use self.embedding? --
        y = np.einsum('bld,nd->bln', x, self.embedding.weight)
        # y = self.output(x)    # -- LLaMA vs embedding.weight ? --
        if(targets is not None):
            # loss = np.mse_loss(x, self.embedding(targets))
            loss = np.cross_entropy(y.view(-1, y.size(-1)), t.reshape(-1), ignore_index=-1)
            return y, loss
        else:
            return np.argmax(y[:,-1:,:], dim=-1)

    def generate(self, prompts : str, max_length : int = 1024):
        prompt_tokens = [self.tokenizer.bos_id()]+self.tokenizer.encode(prompts)
        print('prompt_tokens', len(prompt_tokens))
        for i in range(len(prompt_tokens)):
            output_token_ids = self(np.array([[prompt_tokens[i]]]))

        response_token_ids = output_token_ids
        for _ in range(max_length):
            output_token_ids = self(output_token_ids)
            response_token_ids = np.cat((response_token_ids, output_token_ids), dim=1)
            if self.tokenizer.eos_id() in output_token_ids:
                break

        response_tokens = response_token_ids.squeeze(0).tolist()
        return self.tokenizer.decode(response_tokens)

    # -- Reference: LLaMA and Gemma， Could be learned automaticlly? --
    def precompute_freqs_cis(dim: int,
                            end: int,
                            theta: float = 10000.0):
        """Precomputes the frequency cis."""
        freqs = 1.0 / (theta**(np.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        t = np.arange(end, device=freqs.device)
        freqs = np.outer(t, freqs).float()
        freqs_cis = np.polar(np.ones_like(freqs), freqs)  # complex64
        return freqs_cis

    freqs_cis = None
    if getattr(args, 'rotary_embedding', False):
        # Pre-compute rotary embedding table.
        rope_theta = getattr(args, 'rope_theta', 10000)
        attn_hidden_dim = getattr(args, 'attn_hidden_dim', args.latent_dim)
        attn_heads = getattr(args, 'attn_heads', 1)
        freqs_cis = precompute_freqs_cis(
                            attn_hidden_dim//attn_heads,
                            args.block_size,
                            theta=rope_theta)
    
    pe = None
    if getattr(args, 'position_embedding', False):
        pe = nn.Parameter(np.rand(args.block_size, args.latent_dim), require_grads=True)

    make_layer = MetaLayer if not hasattr(args, 'MetaLayer') else args.MetaLayer
    return nn.Module(
        forward = forward,
        generate = generate,
        tokenizer = args.tokenizer,
        block_size = args.block_size,
        embedding = nn.Embedding(num_embeddings=args.vocab_size, embedding_dim=args.latent_dim),
        layers = nn.ModuleList([make_layer(key, args) for key in args.layers]),
        post_norm = RMSNorm(args.latent_dim),
        prev_norm = getattr(args, 'prev_norm', False),
        # output = nn.Linear(args.latent_dim, args.vocab_size, bias=args.bias), # LLaMA
        pe = pe,
        freqs_cis = freqs_cis
    )
