from ann import np

def split_data_set(self, input_data: np.ndarray, label_index: int, 
                training_set_size: float, validation_set_size: float, normalize: bool) \
                -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        
        # Splits input data into labels and data
        labels = input_data[:, label_index]  # type: ignore
        data_set: np.ndarray = np.delete(input_data, label_index, 1)

        # Normalizes data
        if normalize:
            data_set = self.normalize_data(data_set)

        # Splits data into sets
        nr_of_rows: int = data_set.shape[0]

        training_index: int = int(np.floor(nr_of_rows * training_set_size))
        validation_index: int = training_index + int(np.floor(nr_of_rows * validation_set_size))
        
        training_set = data_set[:training_index, :]
        training_labels = labels[:training_index]

        validation_set = data_set[training_index:validation_index, :]
        validation_labels = labels[training_index:validation_index]

        testing_set = data_set[validation_index:, :]
        testing_labels = labels[validation_index:]

        return training_set, training_labels, validation_set, validation_labels, testing_set, testing_labels