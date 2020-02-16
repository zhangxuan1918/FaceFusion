def generate_tf_record_from_data_file(
        processor,
        input_data_dir,
        train_data_set_name,
        test_data_set_name,
        train_data_output_path,
        test_data_output_path
    ):
    assert input_data_dir and train_data_set_name
    train_input_data_examples = processor.get_train_examples(
        data_dir, train_data_set_name
    )
    convert_examples_to_features(train_input_data_examples)

    num_training_data = len(train_input_data_examples)
