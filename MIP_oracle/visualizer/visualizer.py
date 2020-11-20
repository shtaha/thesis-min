import json

import numpy as np


def format_matrix(matrix, spacing=None, decimals=4):
    lines = []
    matrix = np.squeeze(matrix)
    matrix = np.atleast_2d(matrix)

    max_value = np.max(np.abs(matrix))
    if not spacing:
        if max_value > 0 and not np.isinf(max_value):
            spacing = max([int(np.log10(max_value)) + 5, 6])
        else:
            spacing = 6

    for row in matrix:
        line = ""
        for cell in row:
            if not np.isinf(np.abs(cell)):
                if cell == 0 or np.abs(int(cell) - cell) < 1e-12:
                    pattern = "{:>" + str(int(spacing)) + "}"
                    line = line + pattern.format(int(cell))
                else:
                    pattern = (
                        "{:>" + str(int(spacing)) + "." + str(int(decimals)) + "f}"
                    )
                    line = line + pattern.format(cell)
            else:
                pattern = "{:>" + str(int(spacing)) + "}"
                line = line + pattern.format(cell)

        lines.append(line)
    return lines


def print_matrix(matrix, name=None, spacing=None, decimals=4):
    if type(matrix) == np.ndarray:
        shape = matrix.shape
    else:
        shape = len(matrix)

    if name:
        print(name, "=", str(shape), str(type(matrix)))
    else:
        print(str(shape), str(type(matrix)))

    lines = format_matrix(matrix, spacing=spacing, decimals=decimals)
    print("\n".join(lines))
    print("\n")


def print_dict(dictionary):
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    print(json.dumps(dictionary, indent=1, cls=NumpyEncoder))


def pprint(*args, shift=40):
    if len(args) < 2:
        raise ValueError("At least two arguments for printing.")

    format_str = (
        "{:<" + str(shift) + "}" + "\t".join(["{}" for _ in range(len(args) - 1)])
    )
    print(format_str.format(*args))
