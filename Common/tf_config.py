import tensorflow as tf

def get_no_gpu():
    config = tf.ConfigProto(device_count = {'GPU': 0}, log_device_placement=True)
    # TODO: is the next line necessary?
    config.gpu_options.per_process_gpu_memory_fraction=0.3 # don't hog all vRAM
    config.operation_timeout_in_ms=50000   # terminate on long hangs
    return config