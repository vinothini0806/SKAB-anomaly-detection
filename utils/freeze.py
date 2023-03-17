
from collections import Iterable

def set_freeze_by_id(model, layer_num_last):
    # model.parameters() -> returns an iterator over all trainable parameters of a neural network model.
    for param in model.parameters():
        #  indicating that its gradient does not need to be computed during backpropagation
        param.requires_grad = False
    # model.children() returns an iterator over the child modules of the given model.
    # These sub-modules/children can themselves be instances of nn.Module, so calling model.children() 
    # multiple times can be used to recursively iterate through all the modules of the model.
    child_list = list(model.children())[-layer_num_last:]
    # isinstance(child_list, Iterable) checks if the child_list object is iterable or not.
    if not isinstance(child_list, Iterable):
        child_list = list(child_list)
    for child in child_list:
        for param in child.parameters():
            param.requires_grad = True