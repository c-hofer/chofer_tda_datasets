from collections import defaultdict


def read_graph_from_metis_file(file_path):
    edges = set()
    vertices = []
    degree = defaultdict(int)

    with open(file_path, 'r') as f:
        header = f.readline()[:-1]
        n_vertices, n_edges = (int(x) for x in header.split(' '))
        for node_id, line in enumerate(f):
            # remove \n at end of line
            line = line[:-1]
            neighbor_ids = line.split(' ')
            neighbor_ids = [int(s) for s in neighbor_ids]

            for neig_id in neighbor_ids:

                edge = [node_id, neig_id]
                edge = sorted(edge)
                edge = tuple(edge)

                len_before_add = len(edges)
                edges.add(edge)

                if len_before_add + 1 == len(edges):
                    degree[edge[0]] += 1
                    degree[edge[1]] += 1

            vertices.append((node_id,))

        assert len(vertices) == n_vertices
        assert len(edges) == n_edges

    degree = [degree[i] for i in range(len(degree))]

    return vertices, list(edges), degree