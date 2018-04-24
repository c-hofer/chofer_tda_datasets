from generation.anon_eigenvalue_predict import run


if __name__ == '__main__':
    def get_graph_id_from_path(path):
        return int(path.name.split('.')[0])

    run(raw_data_dir_name='anon_50k_eigenvalue_predict',
        get_graph_id_from_path=get_graph_id_from_path,
        graph_file_extension='metis',
        eigenvalue_file_extension='ev',
        output_file_name='anon_50k_eigenvalue_predict_pershom_degree_filtration.h5',
        max_cpu=10)