Selected filters: basic,comments,stars,fertility
05/09/2025 13:37:14 - INFO - __main__ - ** The job is running with the following arguments: **
Namespace(dataset_name='bigcode/the-stack-dedup', subset='data/java', split='train', tokenizer_name='bigcode/digit-bytelevel-bpe-jss-v1.1-49152', line_max=1000, line_mean=100, alpha_frac=0.25, min_threshold_comments=0.01, max_threshold_comments=0.8, threshold_stars=5, min_size=100, max_size=5000, per_extension_filter_csv=None, num_workers=96, batch_size=1000, push_to_hub=False, remote_repo='test_filter_pipeline_java', hub_username='loubnabnl', out_path=None, log_file='filtering.log', fix_license_columns=False, run_decontamination=False, filters='basic,comments,stars,fertility')
 **** 
05/09/2025 13:37:14 - INFO - __main__ -  ===== Selected filters: ['basic', 'fertility', 'stars', 'comments']=====
05/09/2025 13:37:14 - INFO - __main__ -  ===== Loading bigcode/the-stack-dedup and subset data/java=====
Traceback (most recent call last):
  File "/content/bigcode-dataset/preprocessing/filtering.py", line 261, in <module>
    dataset = load_dataset(
              ^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/datasets/load.py", line 1718, in load_dataset
    builder_instance = load_dataset_builder(
                       ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/datasets/load.py", line 1488, in load_dataset_builder
    dataset_module = dataset_module_factory(
                     ^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/datasets/load.py", line 1216, in dataset_module_factory
    raise e1 from None
  File "/usr/local/lib/python3.11/dist-packages/datasets/load.py", line 1202, in dataset_module_factory
    ).get_module()
      ^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/datasets/load.py", line 766, in get_module
    else get_data_patterns_in_dataset_repository(hfh_dataset_info, self.data_dir)
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/datasets/data_files.py", line 675, in get_data_patterns_in_dataset_repository
    return _get_data_files_patterns(resolver)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/datasets/data_files.py", line 236, in _get_data_files_patterns
    data_files = pattern_resolver(pattern)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/datasets/data_files.py", line 486, in _resolve_single_pattern_in_dataset_repository
    glob_iter = [PurePath(filepath) for filepath in fs.glob(PurePath(pattern).as_posix()) if fs.isfile(filepath)]
                                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/fsspec/spec.py", line 611, in glob
    pattern = glob_translate(path + ("/" if ends_with_sep else ""))
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/fsspec/utils.py", line 731, in glob_translate
    raise ValueError(
ValueError: Invalid pattern: '**' can only be an entire path component


