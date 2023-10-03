from typing import Optional
import numpy as np


class Graph:
    def __init__(self, matrix: Optional[np.ndarray] = None, dtype=np.int64):
        self.matrix = matrix if matrix != None else np.array([])
        self.dtype = dtype

    def __str__(self) -> str:
        return str(self.matrix)

    def from_file(self, file: str, divider: str = ' '):
        with open(file) as f:
            lines = [line.strip().split(divider) for line in f.readlines()]
            self.matrix = np.array(lines).astype(self.dtype)

    def pow(self, mat: np.ndarray) -> np.ndarray:
        result = np.zeros(mat.shape, dtype=self.dtype)

        for row in range(0, mat.shape[0]):
            for col in range(0, mat.shape[1]):
                
                element = mat[row][col]
                if element != 0:
                    result[row] += (self.matrix[col]*element)

        return result
