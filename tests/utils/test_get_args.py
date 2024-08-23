from utils.arg_names import get_args_names


def test_get_args_no_args():
    def no_args_function():
        pass

    assert get_args_names(no_args_function) == []


def test_get_args_single_arg():
    def single_arg_function(arg1):
        pass

    assert get_args_names(single_arg_function) == ["arg1"]


def test_get_args_multiple_args():
    def multiple_args_function(arg1, arg2, arg3):
        pass

    assert get_args_names(multiple_args_function) == ["arg1", "arg2", "arg3"]


def test_get_args_with_defaults():
    def with_defaults(arg1, arg2="default", arg3=10):
        pass

    assert get_args_names(with_defaults) == ["arg1", "arg2", "arg3"]
