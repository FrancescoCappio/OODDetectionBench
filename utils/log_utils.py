import time 

def count_parameters(model):
    total_params = 0
    for _, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        total_params+=param
    return total_params


class LogUnbuffered:

    def __init__(self, args, stream, out_file=None):
        self.args = args
        if self.args.distributed and self.args.global_rank > 0:
            self.out_file = None
            return

        self.stream = stream
        if out_file:
            self.out_file = open(out_file, 'a')
        else:
            self.out_file = None

    def write(self, data):
        if self.args.distributed and self.args.global_rank > 0:
            return
        self.stream.write(data)
        if self.out_file:
            self.out_file.write(data)    # Write the data of stdout here to a text file as well
        self.flush()
    
    def flush(self):
        if self.args.distributed and self.args.global_rank > 0:
            return
        self.stream.flush()
        if self.out_file:
            self.out_file.flush()

    def close(self):
        if self.out_file:
            self.out_file.close()


class CompProfiler:
    """A profiler to estimate time necessary to perform a comparison with a given evaluator
    """

    def __init__(self):
        self.comps_count = 0
        self.total_time = 0

    def start(self):
        """Mark start of comparison"""
        self.start_time = time.time()

    def end(self):
        """Mark end of comparison"""
        self.total_time += time.time() - self.start_time
        self.comps_count += 1

    def __str__(self):
        return f"Total comps {self.comps_count}, Total comp time {self.total_time}. Average comparison time {self.total_time/self.comps_count}"
