def assert_and_raise(obj, obj_type):
    if isinstance(obj, obj_type):
        pass
    else:
        raise TypeError(f"input must be of type `{obj_type.__name__}` got {type(obj).__name__}")
