import copy
import functools
import types


def copy_func_with_new_globals(f, globals=None):
    """Based on https://stackoverflow.com/a/13503277/2988730 (@unutbu)"""
    if globals is None:
        globals = f.__globals__
    g = types.FunctionType(f.__code__, globals, name=f.__name__,
                           argdefs=f.__defaults__, closure=f.__closure__)
    g = functools.update_wrapper(g, f)
    g.__module__ = f.__module__
    g.__kwdefaults__ = copy.copy(f.__kwdefaults__)
    return g


def add_wrapper_after_function_call_in_method(module, method_name, function_name, wrapper_fn):
    '''
    This function adds a wrapper after the output of a function call in the method named `method_name`. 
    Only calls directly in the method are affected. Calls by other functions called in the method are not affected.
    '''

    original_method = getattr(module, method_name).__func__
    method_globals = dict(original_method.__globals__)
    
    # Handle case where function is not in method globals
    if function_name not in method_globals:
        # Try to import from the parent module for known functions
        if function_name == 'apply_rotary_pos_emb':
            # Import from appropriate module based on model type
            if 'llama' in module.__class__.__name__.lower():
                from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
                method_globals[function_name] = apply_rotary_pos_emb
            elif 'mistral' in module.__class__.__name__.lower():
                from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb
                method_globals[function_name] = apply_rotary_pos_emb
            elif 'qwen' in module.__class__.__name__.lower():
                from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
                method_globals[function_name] = apply_rotary_pos_emb
            else:
                raise KeyError(f"Function '{function_name}' not found in method globals and no fallback available")
        else:
            raise KeyError(f"Function '{function_name}' not found in method globals")
    
    wrapper = wrapper_fn(method_globals[function_name])
    method_globals[function_name] = wrapper
    new_method = copy_func_with_new_globals(original_method, globals=method_globals)
    setattr(module, method_name, new_method.__get__(module))
    return wrapper
