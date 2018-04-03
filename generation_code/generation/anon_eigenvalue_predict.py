import multiprocessing

import h5py
import numpy as np
from pershombox import toplex_persistence_diagrams

from .utils.graph import read_graph_from_metis_file
from .path_config import data_raw_path, data_generated_path
from .utils.gui import SimpleProgressCounter


def job_args_list(raw_data_dir,
                  get_graph_id_from_path,
                  graph_file_extension='metis',
                  eigenvalue_file_extension='eigenvalues'):
    gr_files = sorted(raw_data_dir.glob('*.' + graph_file_extension))
    ev_files = sorted(raw_data_dir.glob('*.' + eigenvalue_file_extension))
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


def degree_filtration(simplex, vertex_degree_list):
    return max(vertex_degree_list[vertex_id] for vertex_id in simplex)


def job(args):
    graph_index = args['graph_index']
    graph_id = args['graph_id']
    graph_file_path = args['graph_file_path']
    ev_file_path = args['ev_file_path']

    vertices, edges, vertex_degree_list = read_graph_from_metis_file(graph_file_path)
    eigenvalues = np.loadtxt(ev_file_path)

    list_of_toplices = vertices + edges
    filtration_values = [degree_filtration(s, vertex_degree_list) for s in list_of_toplices]
    dgms_by_dim = toplex_persistence_diagrams(list_of_toplices, filtration_values, deessentialize=False)

    dim_0 = [(b, d) for b, d in dgms_by_dim[0] if d != float('inf')]
    dim_0_ess = [b for b, d in dgms_by_dim[0] if d == float('inf')]
    dim_1_ess = [b for b, d in dgms_by_dim[1] if d == float('inf')]

    ret_val = {'graph_index': graph_index,
               'graph_id': graph_id,
               'dim_0': np.array(dim_0),
               'dim_0_ess': np.array(dim_0_ess),
               'dim_1_ess': np.array(dim_1_ess),
               'eigenvalues': eigenvalues}

    return ret_val


def run(raw_data_dir_name,
        get_graph_id_from_path,
        graph_file_extension,
        eigenvalue_file_extension,
        output_file_name,
        read_me_txt="",
        max_cpu=10):
    raw_data_dir = data_raw_path.joinpath(raw_data_dir_name)
    output_path = data_generated_path.joinpath(output_file_name)

    job_args = job_args_list(raw_data_dir,
                             get_graph_id_from_path=get_graph_id_from_path,
                             graph_file_extension=graph_file_extension,
                             eigenvalue_file_extension=eigenvalue_file_extension)

    progress = SimpleProgressCounter(len(job_args))
    progress.display()
    n_cores = min(multiprocessing.cpu_count() - 1, max_cpu)

    with h5py.File(output_path, 'w') as h5file:

        grp_data = h5file.create_group('data')
        ds_target = h5file.create_dataset('target',
                                          dtype=h5py.special_dtype(vlen=float),
                                          shape=(len(job_args),))

        ds_index_to_id = h5file.create_dataset('index_to_id',
                                               dtype=int,
                                               shape=(len(job_args),))

        ds_read_me = h5file.create_dataset('readme', (1,), dtype=h5py.special_dtype(vlen=str))

        ds_read_me[0] = read_me_txt

        with multiprocessing.Pool(n_cores) as p:

            for ret_val in p.imap_unordered(job, job_args):
                index = ret_val['graph_index']
                graph_id = ret_val['graph_id']
                dim_0 = ret_val['dim_0']
                dim_0_ess = ret_val['dim_0_ess']
                dim_1_ess = ret_val['dim_1_ess']
                eigenvalues = ret_val['eigenvalues']

                grp_index = grp_data.create_group(str(index))

                grp_index.create_dataset('dim_0', data=dim_0)
                grp_index.create_dataset('dim_0_ess', data=dim_0_ess)
                grp_index.create_dataset('dim_1_ess', data=dim_1_ess)

                ds_target[index] = eigenvalues
                ds_index_to_id[index] = graph_id

                progress.trigger_progress()
