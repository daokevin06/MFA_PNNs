import ast

def is_unimodal(data):
    """
    Input data is a string of form e.g. (2,3,4,5,4,6,4,1).
    Output is True if the sequence is unimodal and False if not.
    """

    # 1. Parse the string into a list safely
    if isinstance(data, str):
        try:
            # ast.literal_eval safely evaluates strings containing Python literals
            data = ast.literal_eval(data)
        except (ValueError, SyntaxError):
            raise ValueError(f"Could not parse the string: '{data}'. Ensure it is formatted like '[1, 2, 3]'.")
            
    # 2. Validate the data type
    if not isinstance(data, list):
        raise TypeError("Input must be a list or a string representation of a list.")
        
    # 3. Core unimodal logic
    n = len(data)
    if n <= 2:
        return True
        
    i = 0
    
    # Phase 1: Walk up the non-decreasing slope
    while i + 1 < n and data[i] <= data[i + 1]:
        i += 1
        
    # Phase 2: Walk down the non-increasing slope
    while i + 1 < n and data[i] >= data[i + 1]:
        i += 1
        
    # Phase 3: Check if we reached the end
    return i == n - 1