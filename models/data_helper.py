from os.path import join
import torch
import torch.utils.data as data
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

PATH_TO_TXT = 'data/txt_lists'
NORMALIZE_STATS = {
        "ImageNet1k": {"mean":(0.485, 0.456, 0.406), "std":(0.229, 0.224, 0.225)},
        "CLIP": {"mean":(0.48145466, 0.4578275, 0.40821073), "std":(0.26862954, 0.26130258, 0.27577711)},
        "BiT": {"mean":(0.5,0.5,0.5), "std":(0.5,0.5,0.5)},
        }
NUM_WORKERS = 4

def get_norm_stats(args): 
    mode = "ImageNet1k"
    if args.model == "BiT" or args.model == "CE-IM22k":
        mode = "BiT"
    elif args.model == "clip":
        mode = "CLIP"

    print(f"Using normalization stats from {mode}")
    return NORMALIZE_STATS[mode]

def few_shot_subsample(names, labels, n_shots=5, seed=42):
    print("Applying few shot subsample")
    # randomly select n_shots samples for each class 
    np_labels = np.array(labels)
    np_names = np.array(names)

    indices = np.arange(len(labels))
    known_classes = set(labels)

    random_generator = np.random.default_rng(seed=seed)
    new_names = []
    new_labels = []
    for lbl in known_classes:
        mask = np_labels == lbl
        selected_ids = random_generator.choice(indices[mask],n_shots)
        new_names.extend(np_names[selected_ids].tolist())
        new_labels.extend(n_shots*[lbl])

    return new_names, new_labels

def _dataset_info_standard(txt_labels, lbl_delta=0):
    with open(txt_labels, 'r') as f:
        images_list = f.readlines()

    file_names = []
    labels = []

    for row in images_list:
        row = row.split(' ')
        file_names.append(row[0])
        labels.append(int(row[1]) + lbl_delta)

    return file_names, labels

def _get_known_class_names(file_names, labels):
    # extract name of known classes from paths of support samples
    # we expect path with structure:
    # /bla/bla/.../bla/class_name/image.jpg
    
    class_names = {}
    for lbl, fp in zip(labels, file_names):
        if not lbl in class_names:
            name = fp.split("/")[-2]
            # remove underscores 
            name = name.replace("_", " ")
            class_names[lbl] = name
            
    sorted_names = [class_names[lbl] for lbl in sorted(class_names.keys())]
    return sorted_names

class Dataset(data.Dataset):
    def __init__(self, names,labels, path_dataset, img_transformer=None):
        self.data_path = path_dataset
        self.names = names
        self.labels = labels
        self._image_transformer = img_transformer

    def __getitem__(self, index):

        framename = self.data_path + '/' + self.names[index]
        img = Image.open(framename).convert('RGB')
        img = self._image_transformer(img)

        return img, int(self.labels[index])

    def __len__(self):
        return len(self.names)

def get_eval_dataloader(args):

    dataset = args.dataset
    test = "out_distribution" if args.test is None else args.test
    support = "in_distribution" if args.support is None else args.support

    if not args.data_order == -1:
        support = support + f"_o{args.data_order}"
        test = test + f"_o{args.data_order}"
    print(f"Dataset {dataset}, train_data: {support}, test_data: {test}")

    train_dataset_txt_path = join(PATH_TO_TXT, dataset, support + '.txt')
    names_supports, labels_supports = _dataset_info_standard(train_dataset_txt_path)

    n_known_classes = len(set(labels_supports))
    # apply few shot subsampling if necessary
    if args.few_shot > 0: 
        names_supports, labels_supports = few_shot_subsample(names_supports, labels_supports, n_shots=args.few_shot, seed=args.seed)

    if dataset == "MCM_benchmarks" and support in ["Stanford-Cars", "CUB-200", "Oxford-Pet", "imagenet"]:
        class_names_file = f"data/txt_lists/MCM_benchmarks/{support}_names.txt"
        print(f"Reading class names from: {class_names_file}")

        with open(class_names_file, "r") as f:
            names = f.readlines()
        known_class_names = [name.strip() for name in names]
    elif dataset in ["SUN", "Stanford_Cars"]:
        class_names_file = "class_names"
        if not args.data_order == -1:
            class_names_file = class_names_file + f"_o{args.data_order}"

        class_names_path = join(PATH_TO_TXT, dataset, class_names_file + '.txt')
        print(f"Reading class names from: {class_names_path}")
        with open(class_names_path, "r") as f:
            names = f.readlines()
        known_class_names = [name.strip() for name in names]
    else: 
        print("Attempt to extract class names from file paths")
        known_class_names = _get_known_class_names(names_supports, labels_supports)

    known_class_names = known_class_names[:n_known_classes]
    print(f"Number of known_classes: {n_known_classes}")
    img_tr = get_val_transformer(args)
    train_dataset = Dataset(names_supports, labels_supports, args.path_dataset, img_transformer=img_tr)

    if dataset == "MCM_benchmarks":
        test_ID_txt_path = join(PATH_TO_TXT, dataset, support + "_test.txt")
        test_OOD_txt_path = join(PATH_TO_TXT, dataset, test + ".txt")
        names_test_ID, labels_test_ID = _dataset_info_standard(test_ID_txt_path)
        names_test_OOD, labels_test_OOD = _dataset_info_standard(test_OOD_txt_path, lbl_delta=n_known_classes)
        names_test = names_test_ID + names_test_OOD
        labels_test = labels_test_ID + labels_test_OOD

    else:
        test_dataset_txt_path = join(PATH_TO_TXT, dataset, test + '.txt')
        names_test, labels_test = _dataset_info_standard(test_dataset_txt_path)

    test_dataset = Dataset(names_test, labels_test, args.path_dataset, img_transformer=img_tr)

    if args.distributed:
        test_distributed_sampler = DistributedSampler(dataset=test_dataset, shuffle=False)
        train_distributed_sampler = DistributedSampler(dataset=train_dataset, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.eval_batch_size, sampler=test_distributed_sampler, num_workers=NUM_WORKERS, pin_memory=True, drop_last=False) 
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.eval_batch_size, sampler=train_distributed_sampler, num_workers=NUM_WORKERS, pin_memory=True, drop_last=False)
    else:
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, drop_last=False) 
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, drop_last=False)

    
    return test_loader, train_loader, known_class_names, n_known_classes

def get_train_dataloader(args):

    dataset = args.dataset
    support = "in_distribution" if args.support is None else args.support
    if not args.data_order == -1:
        support = support + f"_o{args.data_order}"

    print(f"Dataset {dataset}, train_data: {support}")

    train_dataset_txt_path = join(PATH_TO_TXT, dataset, support + '.txt')

    names_supports, labels_supports = _dataset_info_standard(train_dataset_txt_path)
    if args.few_shot > 0: 
        names_supports, labels_supports = few_shot_subsample(names_supports, labels_supports, n_shots=args.few_shot, seed=args.seed)

    img_tr = get_train_transformer(args)
    train_dataset = Dataset(names_supports, labels_supports, args.path_dataset, img_transformer=img_tr)

    drop_last = True
    if len(train_dataset) < args.train_batch_size:
        print("Warning: you are training with a dataset smaller than a single batch")
        drop_last = False

    if args.distributed:
        train_distributed_sampler = DistributedSampler(dataset=train_dataset, shuffle=True)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, sampler=train_distributed_sampler, num_workers=NUM_WORKERS, pin_memory=True, drop_last=drop_last)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, drop_last=drop_last)

    return train_loader

def split_train_loader(train_loader, seed):

    whole_dataset = train_loader.dataset
    total_len = len(whole_dataset)
    train_split_len = int(0.8*total_len)
    val_split_len = total_len - train_split_len

    train_set, val_set = torch.utils.data.random_split(whole_dataset, (train_split_len, val_split_len), generator=torch.Generator().manual_seed(seed))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_loader.batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, drop_last=train_loader.drop_last)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=train_loader.batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, drop_last=train_loader.drop_last)

    return train_loader, val_loader


def get_val_transformer(args):

    norm_stats = get_norm_stats(args)

    img_tr = [transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor(),
              transforms.Normalize(mean=norm_stats["mean"], std=norm_stats["std"])]

    return transforms.Compose(img_tr)

def get_train_transformer(args):

    # train transform params 
    min_scale = 0.8
    max_scale = 1.0
    random_horiz_flip = 0.5
    jitter = 0.4
    random_grayscale = 0.1
    norm_stats = get_norm_stats(args)

    img_tr = []

    img_tr.append(transforms.RandomResizedCrop((int(args.image_size), int(args.image_size)), (min_scale, max_scale)))

    if random_horiz_flip > 0.0:
        img_tr.append(transforms.RandomHorizontalFlip(random_horiz_flip))
    if jitter > 0.0:
        img_tr.append(transforms.ColorJitter(brightness=jitter, contrast=jitter, saturation=jitter,hue=min(0.5, jitter)))
    if random_grayscale:
        img_tr.append(transforms.RandomGrayscale(random_grayscale))

    img_tr = img_tr + [transforms.ToTensor(), 
                        transforms.Normalize(norm_stats["mean"], std=norm_stats["std"])]

    return transforms.Compose(img_tr)

def check_data_consistency(train_loader, inference_train_loader):
    # we need to be sure that data used for training and later used as support set at inference time really match 

    train_names = train_loader.dataset.names.copy()
    inf_names = inference_train_loader.dataset.names.copy()

    train_names.sort()
    inf_names.sort()

    for tn, ifn in zip(train_names, inf_names):
        assert tn == ifn, "Datasets do not match"

    print("Train data matches inference train data")

