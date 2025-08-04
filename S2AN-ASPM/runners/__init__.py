from .a3c_train import a3c_train,a3c_train_seen
from .a3c_val import a3c_val,a3c_val_unseen,a3c_val_seen


trainers = [ 
    'vanilla_train',
    'learned_train',
]

testers = [
    'vanilla_val',
    'learned_val',
]

variables = locals()