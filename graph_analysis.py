import sys
from typing import Optional
import numpy as np
from functools import reduce
from graph import Graph


class GraphAnalysis:

    def __init__(self, graph: Graph) -> None:
        self.graph = graph
        self.reachability_matrix, self.matrix_pows = self._reachability_matrix()

    def _reachability_matrix(self) -> tuple[np.ndarray, list[np.ndarray]]:
        """Поиск матрицы достижимости и истории возведения матриц в степень"""

        loop_fallback_counter = 1000
        pows_history = [self.graph.matrix]

        while np.any(pows_history[-1]) and loop_fallback_counter > 0:
            pow_result = self.graph.pow(pows_history[-1])
            pows_history.append(pow_result)
            loop_fallback_counter -= 1

        if loop_fallback_counter == 0:
            raise Exception('Reachability matrix inf loop')

        reachability_matrix = reduce(lambda x, y: x + y, pows_history,
                                     np.zeros(self.graph.matrix.shape)).astype(self.graph.dtype)

        pows_history.insert(0, np.ones(
            self.graph.matrix.shape).astype(self.graph.dtype))

        return (reachability_matrix, pows_history)

    def elements_order(self) -> dict:
        """Определение порядка элементов"""

        order = {}

        for i in range(len(self.matrix_pows)-1):
            sigma_first = np.where(np.any(self.matrix_pows[i], axis=0))[0]
            sigma_second = np.where(
                np.all(self.matrix_pows[i+1] == 0, axis=0))[0]
            order[i] = np.intersect1d(sigma_first, sigma_second)

        return order

    def tact(self, order: dict):
        """Определение тактности системы"""

        return max(order.keys())

    def is_loops(self, *mats: np.ndarray) -> bool:
        """Проверка на существование контуров"""

        for mat in mats:
            for i in range(mat.shape[0]):
                if mat[i][i] != 0:
                    return True

        return False

    def input_nodes(self) -> np.ndarray:
        """Определение входных элементов"""

        return np.where(np.all(self.graph.matrix == 0, axis=0))[0]

    def output_nodes(self) -> np.ndarray:
        """Определение выходных элементов"""

        return np.where(np.all(self.graph.matrix.transpose() == 0, axis=0))[0]

    def hanging_nodes(self) -> np.ndarray:
        """Определение висящих элементов"""

        nodes = []

        for i in range(self.graph.matrix.shape[0]):
            if np.sum(self.graph.matrix[i]) == 0 and np.sum(self.graph.matrix[:, i]) == 0:
                nodes.append(i)

        return np.array(nodes)

    def paths_count(self, from_n: int, to_n: int, length: Optional[int] = None) -> int:
        """Определение путей"""
        
        from_n -= 1
        to_n -= 1

        mat = self.matrix_pows[length] if length else self.reachability_matrix

        if 0 <= from_n < mat.shape[0] and 0 <= to_n < mat.shape[1]:
            return mat[from_n][to_n]

        return -1
    
    def all_paths(self, length: Optional[int] = None) -> list[tuple[int,int,int]]:
        """Определение всех путей
        
        [0] - 1 вершина,
        [1] - 2 вершина,
        [3] - кол-во путей
        """

        result = []

        for i in range(self.graph.matrix.shape[0]):
            for j in range(self.graph.matrix.shape[1]):
                count = self.paths_count(i+1, j+1, length)
                if count:
                    result.append((i,j,count))

        return result

    def formation_involved_elements(self, node: int) -> tuple[dict[int, int], dict[int, int]]:
        """Определение всех элементов, участвующих в формировании данного

        [0] - участвующие в формировании node (K), + сколько раз используется (V)

        [1] - в формировании которых используется node (K) + сколько раз используется (V)

        """

        def _involved_formation_helper(matrix: np.ndarray, node: int) -> dict[int, int]:
            nodes = np.where(matrix[node-1] != 0)[0]
            usage = np.take(matrix[node-1], nodes)

            nodes_with_usage = {}
            # for i in range(nodes.shape[0]):
            #     nodes_with_usage[nodes[i]] = usage[i]
            for n, u in zip(nodes, usage):
                nodes_with_usage[n] = u
            return nodes_with_usage

        previous_nodes = _involved_formation_helper(
            self.reachability_matrix.transpose(), node)
        next_nodes = _involved_formation_helper(self.reachability_matrix, node)

        return (previous_nodes, next_nodes)


class GraphAnalysisWriter:
    def __init__(self, analyzer: GraphAnalysis, file: str) -> None:
        self.analyzer = analyzer
        self.console = sys.stdout
        self.file = open(file, 'w', encoding="utf-8")

    def w(self, object):
        self.console.write(f'{object}\n')
        self.file.write(f'{object}\n')

    def f(self):
        self.console.flush()
        self.file.flush()
        self.file.close()

    def write_analysis(self):
        self.w('Матрица достижимости')
        self.w(self.analyzer.reachability_matrix)
        self.w('\n')

        self.w('Степени матрицы')
        self.w('\n\n'.join([mat.__str__()
               for mat in self.analyzer.matrix_pows]))
        self.w('\n')

        self.w('Порядок элементов')
        order = self.analyzer.elements_order()
        self.w('\n'.join([f'{k} - {v+1}' for k, v in order.items()]))
        self.w('\n')

        self.w('Тактность системы')
        self.w(self.analyzer.tact(order))
        self.w('\n')

        self.w('Наличие контуров')
        self.w(self.analyzer.is_loops(self.analyzer.reachability_matrix))
        self.w('\n')

        self.w('Входные элементы')
        self.w(self.analyzer.input_nodes()+1)
        self.w('\n')

        self.w('Выходные элементы')
        self.w(self.analyzer.output_nodes()+1)
        self.w('\n')

        self.w('Висящие вершины')
        self.w(self.analyzer.hanging_nodes()+1)
        self.w('\n')

        length = 3
        self.w(f'Количество путей длиной {length}')
        self.w('\n'.join([f'{n1+1}-{n2+1} = {count}' for n1, n2, count in self.analyzer.all_paths(length)]))
        self.w('\n')

        self.w(f'Количество путей')
        self.w('\n'.join([f'{n1+1}-{n2+1} = {count}' for n1, n2, count in self.analyzer.all_paths()]))
        self.w('\n')

        node = 5
        self.w(f'Вершины, участвующие в формировании вершины {node}')
        involved_nodes = self.analyzer.formation_involved_elements(node)
        self.w(', '.join([f'X{k+1}: {v} раз' for k,
               v in involved_nodes[0].items()]))
        self.w(', '.join([f'X{k+1}: {v} раз' for k,
               v in involved_nodes[1].items()]))

        self.f()
