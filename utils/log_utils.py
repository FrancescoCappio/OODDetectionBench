
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

