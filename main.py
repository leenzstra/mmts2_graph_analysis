from graph import Graph
from graph_analysis import GraphAnalysis, GraphAnalysisWriter

file = './inputs/task2.txt'
output_file = 'output.txt'

if __name__ == '__main__':
    graph = Graph()
    graph.from_file(file)

    analyzer = GraphAnalysis(graph)
    helper = GraphAnalysisWriter(analyzer, output_file)
    helper.write_analysis()
