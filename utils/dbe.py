import inspect
import os

# Typehinting
from os import PathLike
from typing import Any, Callable

# Global variable to control printing of caller details
PRINT_CALLER_DETAILS = False

# Global vairable to control if it should print on calls where it should not exit
PRINT_WHEN_NOT_EXIT = False

# Global variable to controll if program should exit after debug printing
EXIT_AFTER_PRINT = True

# Global variable to controll if the function should print and exit or do nothing
PRINT_AND_EXIT = True

# Global counter and limit for the number of calls before exiting
CALL_COUNT = 0
CALL_LIMIT = 1  # Set this to the X number of times you want to call the function


def list_files(directory: PathLike) -> bool:
    return os.listdir(directory)


def file_exists(filepath: PathLike) -> bool:
    return os.path.isfile(filepath)


def dbe(
        *args: Any, 
        should_exit: bool = EXIT_AFTER_PRINT, 
        print_caller_details: bool = PRINT_CALLER_DETAILS, 
        calls_before_exit: int = CALL_LIMIT, 
        print_when_not_exit: bool = PRINT_WHEN_NOT_EXIT,
        print_and_exit: bool = PRINT_AND_EXIT,
        eval: Callable[[], bool] = None
    ):
    """
    This function, 'dbe' (Debug and Exit), is designed for debugging purposes. 
    It prints the names and values of input variables and can conditionally exit 
    the program based on the provided arguments.

    Parameters:
    *args (Any): 
        A variable number of arguments of any type. The function prints the names 
        and values of these arguments. It requires at least one argument.

    should_exit (bool, optional): 
        A flag to control whether the program should exit after printing. 
        Default is set by the global variable 'EXIT_AFTER_PRINT'.

    print_caller_details (bool, optional): 
        If True, the function prints details of the caller. 
        Default is set by the global variable 'PRINT_CALLER_DETAILS'.

    calls_before_exit (int, optional): 
        The number of calls to this function before the program exits, 
        if 'should_exit' is True. Default is set by the global variable 'CALL_LIMIT'.

    print_when_not_exit (bool, optional): 
        Controls if the function should print the debug information in calls 
        where it is not set to exit. Default is set by the global variable 
        'PRINT_WHEN_NOT_EXIT'.

    print_and_exit (bool, optional): 
        Determines if the function should perform its operations. Default is 
        set by the global variable 'PRINT_AND_EXIT'.

    eval (Callable[[], bool], optional): 
        A lambda function or callable that returns a boolean, used to determine 
        whether to print and exit. Overrides 'print_and_exit' if provided.

    Raises:
    ValueError: If no arguments are passed to the function.

    Note:
    This function modifies a global variable 'CALL_COUNT', which keeps 
    track of the number of times it has been called.
    """
    global CALL_COUNT
    CALL_COUNT += 1

    # Determine if print and exit should be overridden by custom evaluation
    if eval is not None:
        print_and_exit = eval()

    if not print_and_exit or CALL_COUNT < calls_before_exit and not print_when_not_exit:
        return  # Do nothing until the CALL_LIMIT is reached
    
    if not args:
        raise ValueError("Function requires at least one argument")

    frame = inspect.currentframe().f_back

    if print_caller_details:
        # Get the name of the calling function
        calling_function_name = frame.f_code.co_name
        # Get the module name and file path
        module_name = frame.f_globals['__name__']
        file_path = frame.f_globals['__file__']

        print(f"Func: {calling_function_name}() in module: {module_name} ({file_path})")

    try:
        code_string = inspect.getframeinfo(frame).code_context[0].strip()
    except IndexError:
        raise ValueError("Function can only be called with variable arguments")

    # Parse out actual variable names
    var_names = code_string[code_string.find("(")+1:-1].split(",")

    print()

    for var_name, arg in zip(var_names, args):
        var_name_strip = var_name.strip()
        if isinstance(arg, list):
            print(f"{var_name_strip}({len(arg)}) = {arg}")
        elif isinstance(arg, dict):
            print(f"{var_name_strip}(keys:{len(arg.keys())}) = {{")
            for k, v in arg.items():
                print(f"\t{k} : {v}")
            print("}")
        elif isinstance(arg, set):
            print(f"{var_name_strip}({len(arg)}) = {arg}")
        elif isinstance(arg, tuple):
            print(f"{var_name_strip}({len(arg)}) = {arg}")
        else:
            print(f"{var_name_strip} = {arg}")

    if should_exit and CALL_COUNT >= calls_before_exit:
        exit()

if __name__ == '__main__':
    test_var = 'test'

    test_array = [1,2,3,4,5,6]

    test_dict = {
        "key1": "value1",
        "key2": "value2",
        "key3": ["value3a", "value3b", "value3c"],
        "key4": {"nested_key1": "nested_value1", "nested_key2": "nested_value2"}
    }

    test_set = {"apple", "banana", "cherry"}


    dbe(test_var, test_array, test_dict, test_set, calls_before_exit=2, print_when_not_exit=True)

    test_tuple = ("apple", "banana", "cherry")

    dbe(test_tuple)

    dbe('not print')