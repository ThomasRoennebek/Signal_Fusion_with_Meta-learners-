import pytest
import torch
import torch.nn as nn

# TODO: Import your MAML class or training functions once implemented
# from master_thesis.maml import MAML, maml_inner_loop, maml_outer_loop

def test_maml_fast_adaptation():
    """
    Verify that the MAML inner loop updates model weights correctly.
    The weights after adaptation should be different from the base weights.
    """
    # dummy_model = nn.Linear(10, 2)
    # base_weights = dummy_model.weight.clone()
    
    # # Perform one step of gradient descent manually or via your function
    # # updated_weights = maml_inner_loop(dummy_model, support_x, support_y, lr=0.01)
    
    # assert not torch.allclose(base_weights, updated_weights), "Weights did not update during adaptation"
    pass

def test_maml_outer_loop_gradients():
    """
    Verify that the outer loop properly collects gradients from the inner loop
    and updates the meta-parameters.
    """
    # dummy_model = MAML()
    # meta_loss = ... # compute over query set after inner loop
    # meta_loss.backward()
    
    # for name, param in dummy_model.named_parameters():
    #     assert param.grad is not None, f"No gradient for {name}"
    pass
