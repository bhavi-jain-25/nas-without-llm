# Copyright (c) Alibaba, Inc. and its affiliates.
# The implementation is also open-sourced by the authors, and available at
# https://github.com/alibaba/lightweight-neural-architecture-search.

work_dir = './save_model/mbv2_ultra_lowacc/'
log_level = 'INFO'  # INFO/DEBUG/ERROR
log_freq = 1000

""" image config """
# Use very small resolution for extremely low accuracy
image_size = 64  # Much smaller for very low accuracy

""" Model config - ULTRA DEGRADED for <40% accuracy """
model = dict(
    type = 'CnnNet',
    structure_info =[\
        {'class': 'ConvKXBNRELU', 'in': 3, 'out': 2, 's': 8, 'k': 1}, \
        {'class': 'ConvKXBNRELU', 'in': 2, 'out': 4, 's': 1, 'k': 1}, \
     ]
)

""" Budget config """
# Very tight budgets for extremely low accuracy
budgets = [
    dict(type = "flops", budget = 1e6),   # EXTREMELY small
    dict(type = "layers",budget = 3),     # MINIMAL layers
]

""" Score config """
# Use simple random scorer for ultra-low accuracy (no LLM involved)
score = dict(type = 'random')

""" Space config """
space = dict(
    type = 'space_k1dwk1',
    image_size = image_size,
)

""" Search config """
# Minimal search for very low accuracy
search = dict(
    minor_mutation = False,
    minor_iter = 100000,
    popu_size = 8,         # MINIMAL population
    num_random_nets = 20,  # VERY few iterations
    sync_size_ratio = 1.0,
    num_network = 1,
) 