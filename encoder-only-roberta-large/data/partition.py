import numpy as np
from scipy.stats import dirichlet
import torch
from torch.utils.data import DataLoader, RandomSampler, random_split, SequentialSampler, Subset
import copy
import os
import logging


def partition(args, train_dataset, test_dataset, logger):
    train_dataloader_list = [copy.deepcopy(1) for _ in range(args.num_clients)]
    test_dataloader_list = [copy.deepcopy(1) for _ in range(args.num_clients)]
    
    n_sample_list = [0 for _ in range(args.num_clients)]

    
    if args.data_partition_method == 'iid':
        subset_size = len(train_dataset) // args.num_clients
        remaining_size = len(train_dataset) - subset_size * args.num_clients
        subset_sizes = [subset_size] * args.num_clients
        for i in range(remaining_size):
            subset_sizes[i] += 1
        subsets = random_split(train_dataset, subset_sizes)
        print('number of samples')
        for i, subset in enumerate(subsets):
            train_sampler = RandomSampler(subset)
            train_dataloader_list[i] = DataLoader(subset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=train_dataset.collate_fn)
            print(f'Client {i}: {len(train_dataloader_list[i].dataset)}')
            n_sample_list[i] = len(train_dataloader_list[i].dataset)

    elif args.data_partition_method == 'dirichlet_quantity':
        num_clients = args.num_clients
        total_samples = len(train_dataset)
        dirichlet_samples = dirichlet.rvs([args.dirichlet_alpha]*num_clients, size=1)
        client_samples = np.round(dirichlet_samples * total_samples).astype(int)
        subset_sizes = client_samples.squeeze()
        diff = sum(subset_sizes) - total_samples
        subset_sizes[-1] -= diff
        assert min(subset_sizes) > 0, "try a larger dirichlet alpha"
        subsets = random_split(train_dataset, subset_sizes)
        print('number of samples')
        for i, subset in enumerate(subsets):
            train_sampler = RandomSampler(subset)
            train_dataloader_list[i] = DataLoader(subset, sampler=train_sampler, batch_size=args.per_gpu_train_batch_size, collate_fn=train_dataset.collate_fn)
            print(f'Client {i}: {len(train_dataloader_list[i].dataset)}')
            n_sample_list[i] = len(train_dataloader_list[i].dataset)

        #######################################################################################
        # test_loader
        #######################################################################################
        total_samples = len(test_dataset)
        client_samples = np.round(dirichlet_samples * total_samples).astype(int)
        subset_sizes = client_samples.squeeze()
        diff = sum(subset_sizes) - total_samples
        subset_sizes[-1] -= diff
        assert min(subset_sizes) > 0, "try a larger dirichlet alpha"
        subsets = random_split(test_dataset, subset_sizes)
        print('number of samples')
        for i, subset in enumerate(subsets):
            test_sampler = SequentialSampler(subset)
            test_dataloader_list.append(DataLoader(subset, sampler=test_sampler, batch_size=args.per_gpu_train_batch_size, collate_fn=test_dataset.collate_fn))
            print(f'Client {i}: {len(test_dataloader_list[i].dataset)}')


    elif args.data_partition_method == 'dirichlet_label':
        labels = np.array(train_dataset.all_labels)
        unique_values = set(labels)
        label_list = list(unique_values)
        if args.load_from_cache and (os.path.exists(os.path.join(args.data_dir, 'saved_train_subsets.pkl'))) \
            and os.path.exists(os.path.join(args.data_dir, 'saved_test_subsets.pkl')):
            logger.info("read from saved train and test subsets")
            train_subsets_list = torch.load(os.path.join(args.data_dir, 'saved_train_subsets.pkl'))
            test_subsets_list = torch.load(os.path.join(args.data_dir, 'saved_test_subsets.pkl'))

            for client_idx in range(args.num_clients):
                subset = train_subsets_list[client_idx]
                train_sampler = RandomSampler(subset)
                train_dataloader_list[client_idx] = DataLoader(subset, sampler=train_sampler, batch_size=args.per_gpu_train_batch_size, collate_fn=train_dataset.collate_fn)
                
            for client_idx in range(args.num_clients):
                subset = test_subsets_list[client_idx]
                test_sampler = SequentialSampler(subset)
                test_dataloader_list[client_idx] = DataLoader(subset, sampler=test_sampler, batch_size=args.per_gpu_eval_batch_size, collate_fn=test_dataset.collate_fn)


        else:
            logger.info("creating new datasets from scratch")
            labels = np.array(train_dataset.all_labels)
            num_class = len(label_list)
            num_clients = args.num_clients
            total_samples = len(train_dataset)
            dirichlet_samples = dirichlet.rvs([5]*num_clients, size=1)
            client_samples = np.round(dirichlet_samples * total_samples).astype(int)
            subset_sizes = client_samples.squeeze()
            diff = sum(subset_sizes) - total_samples
            subset_sizes[-1] -= diff
            assert min(subset_sizes) > 0, "try a larger dirichlet alpha"
            num_data_per_client = subset_sizes
            cls_priors   = np.random.dirichlet(alpha=[args.dirichlet_alpha]*num_class,size=num_clients)
            prior_cumsum = np.cumsum(cls_priors, axis=1)
            idx_list = [np.where(labels == i)[0] for i in label_list]
            cls_amount = [len(idx_list[i]) for i in list(range(len(idx_list)))]
            sample_idx_per_client = [[] for _ in range(num_clients)]
            while(np.sum(subset_sizes)!=0):
                curr_clnt = np.random.randint(num_clients)
                if num_data_per_client[curr_clnt] <= 0:
                    continue
                num_data_per_client[curr_clnt] -= 1
                curr_prior = prior_cumsum[curr_clnt]
                while True:
                    cls_label = np.argmax(np.random.uniform() <= curr_prior)
                    # Redraw class label if trn_y is out of that class
                    if cls_amount[cls_label] <= 0:
                        continue
                    cls_amount[cls_label] -= 1
                    sample_idx_per_client[curr_clnt].append(idx_list[cls_label][cls_amount[cls_label]])
                    break
            subset_list = []
            for client_idx in range(num_clients):
                subset = Subset(train_dataset, sample_idx_per_client[client_idx])
                subset_list.append(subset)
                train_sampler = RandomSampler(subset)
                train_dataloader_list[client_idx] = DataLoader(subset, sampler=train_sampler, batch_size=args.per_gpu_train_batch_size, collate_fn=train_dataset.collate_fn)
            torch.save(subset_list, os.path.join(args.data_dir, 'saved_train_subsets.pkl'))
            
            #######################################################################################
            # test_loader
            #######################################################################################
            if args.personalization:
                labels = np.array(test_dataset.all_labels)
                num_class = len(label_list)
                num_clients = args.num_clients
                total_samples = len(test_dataset)
                client_samples = np.round(dirichlet_samples * total_samples).astype(int)
                subset_sizes = client_samples.squeeze()
                diff = sum(subset_sizes) - total_samples
                subset_sizes[-1] -= diff
                assert min(subset_sizes) > 0, "try a larger dirichlet alpha"
                num_data_per_client = subset_sizes
                idx_list = [np.where(labels == i)[0] for i in label_list]
                cls_amount = [len(idx_list[i]) for i in list(range(len(idx_list)))]
                sample_idx_per_client = [[] for _ in range(num_clients)]
                while(np.sum(subset_sizes)!=0):
                    curr_clnt = np.random.randint(num_clients)
                    if num_data_per_client[curr_clnt] <= 0:
                        continue
                    num_data_per_client[curr_clnt] -= 1
                    curr_prior = prior_cumsum[curr_clnt]
                    while True:
                        cls_label = np.argmax(np.random.uniform() <= curr_prior)
                        if cls_amount[cls_label] <= 0:
                            continue
                        cls_amount[cls_label] -= 1
                        sample_idx_per_client[curr_clnt].append(idx_list[cls_label][cls_amount[cls_label]])
                        break
                
                torch.save(test_dataset, os.path.join(args.data_dir, 'saved_testset.pkl'))

                subset_list = []
                for client_idx in range(num_clients):
                    subset = Subset(test_dataset, sample_idx_per_client[client_idx])
                    subset_list.append(subset)
                    test_sampler = SequentialSampler(subset)
                    test_dataloader_list[client_idx] = DataLoader(subset, sampler=test_sampler, batch_size=args.per_gpu_eval_batch_size, collate_fn=test_dataset.collate_fn)
                torch.save(subset_list, os.path.join(args.data_dir, 'saved_test_subsets.pkl'))






    else:
        raise NotImplementedError()


    
    # check data distribution on each client
    
    print("training loaders:")
    for i, loader in enumerate(train_dataloader_list):
        if len(label_list) == 2:
            labels = []
            for batch in loader:
                labels.extend(batch[4])
            l0 = 0
            l1 = 0
            for ids in labels:
                if ids == label_list[0]:
                    l0 += 1
                elif ids == label_list[1]:
                    l1 += 1
                else:
                    raise ValueError()
            print(f"client {i}: label 0 : {l0} label 1 : {l1}")
        n_sample_list[i] = len(train_dataloader_list[i].dataset)
    if args.personalization:
        print("test loaders:")
        for i, loader in enumerate(test_dataloader_list):
            if len(label_list) == 2:
                labels = []
                for batch in loader:
                    labels.extend(batch[4])
                l0 = 0
                l1 = 0
                for ids in labels:
                    if ids == label_list[0]:
                        l0 += 1
                    elif ids == label_list[1]:
                        l1 += 1
                    else:
                        raise ValueError()
                print(f"client {i}: label 0 : {l0} label 1 : {l1}")
        
    return train_dataloader_list, test_dataloader_list, n_sample_list