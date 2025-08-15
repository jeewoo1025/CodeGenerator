def timeout_handler(_, __):
    """Signal handler to raise TimeoutError when a timeout occurs."""
    raise TimeoutError()


import os, json

def to_jsonl(dict_data, file_path):
    """
    Append a dictionary as a JSON line to a file.
    
    Args:
        dict_data (dict): Dictionary to be written as a JSON line.
        file_path (str): Path to the target .jsonl file.
    
    Notes:
        - Uses 'a' mode to append new lines to the file.
        - Each dictionary is stored in one line (JSONL format).
    """
    with open(file_path, 'a') as file:
        json_line = json.dumps(dict_data)
        file.write(json_line + os.linesep)


from threading import Thread

class PropagatingThread(Thread):
    """
    A custom Thread subclass that propagates exceptions to the main thread.
    
    - Stores exceptions raised in the thread and re-raises them when joined.
    - Captures the return value of the target function for later retrieval.
    """

    def run(self):
        self.exc = None
        try:
            # Support for Python < 3.0 where Thread target is name-mangled
            if hasattr(self, '_Thread__target'):
                self.ret = self._Thread__target(*self._Thread__args, **self._Thread__kwargs)
            else:
                self.ret = self._target(*self._args, **self._kwargs)
        except BaseException as e:
            self.exc = e

    def join(self, timeout=None):
        """
        Join the thread and re-raise any exception that occurred in the thread.
        
        Args:
            timeout (float, optional): Maximum number of seconds to wait for the thread.
        
        Returns:
            Any: Return value from the target function.
        
        Raises:
            BaseException: Re-raises the exception from the thread if occurred.
        """
        super(PropagatingThread, self).join(timeout)
        if self.exc:
            raise self.exc
        return self.ret


def function_with_timeout(func, args, timeout):
    """
    Execute a function with a timeout limit using threading.
    
    Args:
        func (callable): The function to execute.
        args (tuple): Arguments to pass to the function.
        timeout (float): Maximum allowed execution time in seconds.
    
    Returns:
        Any: Result of the function call if completed within timeout.
    
    Raises:
        TimeoutError: If the function execution exceeds the timeout.
    """
    result_container = []

    def wrapper():
        result_container.append(func(*args))

    thread = PropagatingThread(target=wrapper)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        raise TimeoutError()
    else:
        return result_container[0]


# Py tests (example usage)
# if __name__ == "__main__":
#     formatter = PySubmissionFormatter()
#     leetcode_1 = 'class Solution:\n    def solveSudoku(self, board: List[List[str]]) -> None:\n        """\n        Do not return anything, modify board in-place instead.\n        """\n        '
#     humaneval_1 = 'def solveSudoku(self, board: List[List[str]]) -> None:\n        """\n        Do not return anything, modify board in-place instead.\n        """\n'
#
#     assert leetcode_1 == formatter.to_leetcode(humaneval_1)
#     assert humaneval_1 == formatter.to_humaneval(leetcode_1)
