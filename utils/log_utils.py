def count_parameters(model):
    total_params = 0
    for _, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        total_params += param
    return total_params


def gen_train_log_msg(it, n_iters, optims_dict, losses_dict, train_acc, log_period):
    log_msg_fields = [f"Iterations: {it+1:6d}/{n_iters}"]

    # get the log and wandb suffixes given the item's name
    get_suffix = lambda name: f"({name})" if name else ""

    # LR
    lr_msg_fields = []
    for optim_name, optim_ in optims_dict.items():
        current_lr = optim_.param_groups[0]["lr"]
        msg_suffix = get_suffix(optim_name)
        lr_msg_fields.append(f"{current_lr:.6f}{msg_suffix}")
    lr_msg = ", ".join(lr_msg_fields)
    log_msg_fields.append(f"LR: {lr_msg}")

    # Loss
    loss_msg_fields = []
    for loss_name, loss_ in losses_dict.items():
        avg_loss = loss_ / log_period
        msg_suffix = get_suffix(loss_name)
        loss_msg_fields.append(f"{avg_loss:6.4f}{msg_suffix}")
    loss_msg = ", ".join(loss_msg_fields)
    log_msg_fields.append(f"Loss: {loss_msg}")

    # Acc
    log_msg_fields.append(f"Acc: {train_acc:6.4f}")

    log_msg = "\t".join(log_msg_fields)

    return log_msg


class LogUnbuffered:
    def __init__(self, args, stream, out_file=None):
        self.args = args
        if self.args.distributed and self.args.global_rank > 0:
            self.out_file = None
            return

        self.stream = stream
        if out_file:
            self.out_file = open(out_file, "a")
        else:
            self.out_file = None

    def write(self, data):
        if self.args.distributed and self.args.global_rank > 0:
            return
        self.stream.write(data)
        if self.out_file:
            self.out_file.write(data)  # Write the data of stdout here to a text file as well
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
