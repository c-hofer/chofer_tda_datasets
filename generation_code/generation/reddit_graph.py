import pickle
import multiprocessing
import h5py
import numpy as np

from collections import defaultdict
from pershombox import toplex_persistence_diagrams
from .utils.gui import SimpleProgressCounter


def load_data(data_set_path):
    with open(data_set_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data


def build_graph(graph_dict):
    edges = set()
    vertices = []
    degree = defaultdict(int)

    for node_id, vertex_dict in graph_dict.items():

        neighbor_ids = vertex_dict['neighbors']

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

    degree = [degree[i] for i in range(len(vertices))]

    return vertices, list(edges), degree


def degree_filtration(simplex, vertex_degree_list):
    return max(vertex_degree_list[vertex_id] for vertex_id in simplex)


def job_args_list(file_path):
    data = load_data(file_path)
    job_args = []

    for graph_id, graph_dict in data['graph'].items():
         graph_id = int(graph_id)
         job_args.append({
                          'graph_id': graph_id,
                          'graph_dict': graph_dict,
                          'label': int(data['labels'][graph_id])
                         })

    return job_args


def job(args):
    graph_id = args['graph_id']
    graph_dict = args['graph_dict']
    label = args['label']

    vertices, edges, vertex_degree_list = build_graph(graph_dict)

    list_of_toplices = vertices + edges
    filtration_values = [degree_filtration(s, vertex_degree_list) for s in list_of_toplices]
    dgms_by_dim = toplex_persistence_diagrams(list_of_toplices, filtration_values, deessentialize=False)

    dim_0 = [(b, d) for b, d in dgms_by_dim[0] if d != float('inf')]
    dim_0_ess = [b for b, d in dgms_by_dim[0] if d == float('inf')]
    dim_1_ess = [b for b, d in dgms_by_dim[1] if d == float('inf')]
    max_degree = max(vertex_degree_list)

    ret_val = {'graph_id': graph_id,
               'dim_0': np.array(dim_0, dtype=float),
               'dim_0_ess': np.array(dim_0_ess, dtype=float),
               'dim_1_ess': np.array(dim_1_ess, dtype=float),
               'label': label,
               'max_degree': float(max_degree)}

    return ret_val


def run(raw_data_path, output_path, max_cpu=10):

    job_args = job_args_list(raw_data_path)

    progress = SimpleProgressCounter(len(job_args))
    progress.display()
    n_cores = min(multiprocessing.cpu_count() - 1, max_cpu)

    with h5py.File(output_path, 'w') as h5file:

        grp_data = h5file.create_group('data')
        ds_target = h5file.create_dataset('target',
                                          dtype=int,
                                          shape=(len(job_args),))

        ds_max_degree = h5file.create_dataset('max_degree',
                                             dtype=float,
                                             shape=(len(job_args),))

        ds_read_me = h5file.create_dataset('readme', (1,), dtype=h5py.special_dtype(vlen=str))
        read_me_txt = \
            """            
            """
        ds_read_me[0] = read_me_txt

        with multiprocessing.Pool(n_cores) as p:

            for ret_val in p.imap_unordered(job, job_args):
                graph_id = ret_val['graph_id']
                dim_0 = ret_val['dim_0']
                dim_0_ess = ret_val['dim_0_ess']
                dim_1_ess = ret_val['dim_1_ess']
                label = ret_val['label']
                max_degree = ret_val['max_degree']

                grp_index = grp_data.create_group(str(graph_id))

                grp_index.create_dataset('dim_0', data=dim_0)
                grp_index.create_dataset('dim_0_ess', data=dim_0_ess)
                grp_index.create_dataset('dim_1_ess', data=dim_1_ess)

                ds_target[graph_id] = label
                ds_max_degree[graph_id] = max_degree

                progress.trigger_progress()
