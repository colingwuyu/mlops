from inspect import signature


def call_func(obj, attr_name, *args, **kwargs):
    callable_func = obj.__getattribute__(attr_name)
    args_list = list(args)
    args_list.reverse()
    callable_func_kwargs = {}
    callable_func_args = []
    for param_name, param in signature(callable_func).parameters.items():
        if param_name == "self":
            continue
        if param.kind == param.POSITIONAL_OR_KEYWORD:
            if param_name in kwargs:
                callable_func_kwargs[param_name] = kwargs[param_name]
                continue
            if param.default is param.empty and len(args_list) > 0:
                callable_func_kwargs[param_name] = args_list.pop()
            else:
                continue
        elif param.kind == param.VAR_POSITIONAL:
            while len(args_list) > 0:
                callable_func_args.append(args_list.pop())
        elif param.kind == param.VAR_KEYWORD:
            callable_func_kwargs.update(kwargs)

    return callable_func(*callable_func_args, **callable_func_kwargs)
