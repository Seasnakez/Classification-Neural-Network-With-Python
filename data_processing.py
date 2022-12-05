from ann import np

def normalize_data(data):
        min_val = np.min(data)
        max_val = np.max(data)
        if min_val == 0:
            return data / max_val
        else:
            return (data - min_val) / (max_val - min_val)
