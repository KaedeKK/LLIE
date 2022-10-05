from .CrossUFormer import Model


def CrossUFormer():
    return Model(
        embed_dims=[24, 48, 96, 96, 48, 24],
		mlp_ratios=[2., 2., 4., 4., 2., 2.],
		depths=[2, 4, 8, 8, 4, 2],
		num_heads=[2, 4, 6, 6, 4, 2],
		attn_ratio=[1/4, 1/2, 3/4, 0, 0, 0],
		conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'])