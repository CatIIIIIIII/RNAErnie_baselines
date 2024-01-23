# sequence length
S = 512
# hidden size
H = 768
# intermediate size
I = H*4
# layer number
L = 12

# self attention
self_attn_size = S*S*H
# ffn
ffn_size = S*H*H*2
# other

print("The number of FLOPs of model {}GFLOPs".format(
    (self_attn_size + ffn_size)*L/10**9))
