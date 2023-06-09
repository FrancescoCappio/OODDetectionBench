from os.path import join
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

PATH_TO_TXT = 'data/txt_lists'
IMAGENET_PIXEL_MEAN = [0.485, 0.456, 0.406]
IMAGENET_PIXEL_STD = [0.229, 0.224, 0.225]
NUM_WORKERS = 4

def few_shot_subsample(names, labels, n_shots=5, seed=42):
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
    # extract name of known classes from paths of source samples
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
    target = "out_distribution" if args.target is None else args.target
    source = "in_distribution" if args.source is None else args.source
    print(f"Dataset {dataset}, train_data: {source}, test_data: {target}")

    train_dataset_txt_path = join(PATH_TO_TXT, dataset, source + '.txt')
    names_sources, labels_sources = _dataset_info_standard(train_dataset_txt_path)
    # apply few shot subsampling if necessary
    if args.few_shot > 0: 
        names_sources, labels_sources = few_shot_subsample(names_sources, labels_sources, n_shots=args.few_shot, seed=args.seed)

    if source in ["Stanford-Cars", "CUB-200", "Oxford-Pet"]:
        class_names_file = f"data/txt_lists/MCM_benchmarks/{source}_names.txt"

        with open(class_names_file, "r") as f:
            names = f.readlines()
        known_class_names = [name.strip() for name in names]
    else: 
        known_class_names = _get_known_class_names(names_sources, labels_sources)
    n_known_classes = len(known_class_names)
    img_tr = get_val_transformer(args)
    train_dataset = Dataset(names_sources, labels_sources, args.path_dataset, img_transformer=img_tr)

    if dataset == "MCM_benchmarks":
        test_ID_txt_path = join(PATH_TO_TXT, dataset, source + "_test.txt")
        test_OOD_txt_path = join(PATH_TO_TXT, dataset, target + ".txt")
        names_test_ID, labels_test_ID = _dataset_info_standard(test_ID_txt_path)
        names_test_OOD, labels_test_OOD = _dataset_info_standard(test_OOD_txt_path, lbl_delta=n_known_classes)
        names_target = names_test_ID + names_test_OOD
        labels_target = labels_test_ID + labels_test_OOD

    else:
        test_dataset_txt_path = join(PATH_TO_TXT, dataset, target + '.txt')
        names_target, labels_target = _dataset_info_standard(test_dataset_txt_path)

    test_dataset = Dataset(names_target, labels_target, args.path_dataset, img_transformer=img_tr)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, drop_last=False) 
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, drop_last=False)
    
    return test_loader, train_loader, known_class_names, n_known_classes

def get_train_dataloader(args):

    dataset = args.dataset
    source = "in_distribution" if args.source is None else args.source
    print(f"Dataset {dataset}, train_data: {source}")

    train_dataset_txt_path = join(PATH_TO_TXT, dataset, source + '.txt')

    names_sources, labels_sources = _dataset_info_standard(train_dataset_txt_path)
    if args.few_shot > 0: 
        names_sources, labels_sources = few_shot_subsample(names_sources, labels_sources, n_shots=args.few_shot, seed=args.seed)

    img_tr = get_train_transformer(args)
    train_dataset = Dataset(names_sources, labels_sources, args.path_dataset, img_transformer=img_tr)

    drop_last = True
    if len(train_dataset) < args.train_batch_size:
        print("Warning: you are training with a dataset smaller than a single batch")
        drop_last = False

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

    img_tr = [transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor(),
              transforms.Normalize(IMAGENET_PIXEL_MEAN, std=IMAGENET_PIXEL_STD)]

    return transforms.Compose(img_tr)

def get_train_transformer(args):

    # train transform params 
    min_scale = 0.8
    max_scale = 1.0
    random_horiz_flip = 0.5
    jitter = 0.4
    random_grayscale = 0.1

    img_tr = []

    img_tr.append(transforms.RandomResizedCrop((int(args.image_size), int(args.image_size)), (min_scale, max_scale)))

    if random_horiz_flip > 0.0:
        img_tr.append(transforms.RandomHorizontalFlip(random_horiz_flip))
    if jitter > 0.0:
        img_tr.append(transforms.ColorJitter(brightness=jitter, contrast=jitter, saturation=jitter,hue=min(0.5, jitter)))
    if random_grayscale:
        img_tr.append(transforms.RandomGrayscale(random_grayscale))

    img_tr = img_tr + [transforms.ToTensor(), transforms.Normalize(IMAGENET_PIXEL_MEAN, std=IMAGENET_PIXEL_STD)]

    return transforms.Compose(img_tr)

def check_data_consistency(train_loader, inference_train_loader):
    # we need to be sure that data used for training and later used as support set at inference time really match 

    train_names = train_loader.dataset.names 
    inf_names = inference_train_loader.dataset.names 

    train_names.sort()
    inf_names.sort()

    for tn, ifn in zip(train_names, inf_names):
        assert tn == ifn, "Datasets do not match"

    print("Train data matches inference train data")

