import multiprocessing
import h5py
import numpy as np

from collections import defaultdict
from pershombox import toplex_persistence_diagrams
from .path_config import data_raw_path, data_generated_path
from .utils.gui import SimpleProgressCounter


def job_args_list(raw_data_dir):
    def get_graph_id_from_path(path):
        return int(path.name.split('-')[0])

    gr_files = sorted(raw_data_dir.glob('*.metis'))
    ev_files = sorted(raw_data_dir.glob('*.eigenvalues'))
    assert len(gr_files) == len(ev_files)
    file_paths = list(zip(gr_files, ev_files))

    job_args = []
    for index, (gr, ev) in enumerate(file_paths):
        gr_id = get_graph_id_from_path(gr)
        assert gr_id == get_graph_id_from_path(ev)
        arg = {'graph_index': index,
               'graph_id': gr_id,
               'graph_file_path': str(gr),
               'ev_file_path': str(ev)}

        job_args.append(arg)

    return job_args


def read_graph_from_file(file_path):
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


def degree_filtration(simplex, vertex_degree_list):
    return max(vertex_degree_list[vertex_id] for vertex_id in simplex)


def job(args):
    graph_index = args['graph_index']
    graph_id = args['graph_id']
    graph_file_path = args['graph_file_path']
    ev_file_path = args['ev_file_path']

    vertices, edges, vertex_degree_list = read_graph_from_file(graph_file_path)
    eigenvalues = np.loadtxt(ev_file_path)

    list_of_toplices = vertices + edges
    filtration_values = [degree_filtration(s, vertex_degree_list) for s in list_of_toplices]
    dgms_by_dim = toplex_persistence_diagrams(list_of_toplices, filtration_values, deessentialize=True)

    ret_val = {'graph_index': graph_index,
               'graph_id': graph_id,
               'barcodes_dim_0': np.array(dgms_by_dim[0]),
               'birth_times_dim_1': np.array(dgms_by_dim[1])[:, 0],
               'eigenvalues': eigenvalues}

    return ret_val


def run():
    raw_data_dir = data_raw_path.joinpath('anon_eigenvalue_predict')
    output_path = data_generated_path.joinpath('anon_eigenvalue_predict_pershom_degree_filtration.h5')

    job_args = job_args_list(raw_data_dir)

    progress = SimpleProgressCounter(len(job_args))
    progress.display()
    n_cores = min(multiprocessing.cpu_count() - 1, 10)

    with h5py.File(output_path, 'w') as h5file:

        grp_data = h5file.create_group('data')
        ds_target = h5file.create_dataset('target',
                                          dtype=h5py.special_dtype(vlen=float),
                                          shape=(len(job_args),))

        ds_index_to_id = h5file.create_dataset('index_to_id',
                                               dtype=int,
                                               shape=(len(job_args),))

        ds_read_me = h5file.create_dataset('readme', (1,), dtype=h5py.special_dtype(vlen=str))
        read_me_txt = \
            """            
            """
        ds_read_me[0] = read_me_txt

        with multiprocessing.Pool(n_cores) as p:

            for ret_val in p.imap_unordered(job, job_args):
                index = ret_val['graph_index']
                graph_id = ret_val['graph_id']
                barcodes_dim_0 = ret_val['barcodes_dim_0']
                birth_times_dim_1 = ret_val['birth_times_dim_1']
                eigenvalues = ret_val['eigenvalues']

                grp_index = grp_data.create_group(str(index))

                grp_index.create_dataset('dim_0', data=barcodes_dim_0)
                grp_index.create_dataset('dim_1', data=birth_times_dim_1)

                ds_target[index] = eigenvalues
                ds_index_to_id[index] = graph_id

                progress.trigger_progress()
