from method.ltsf_linear.LTSF_Linear.data_provider.data_loader import Dataset_Custom
from torch.utils.data import DataLoader

data_dict = {
    'custom': Dataset_Custom,
}


def data_provider(args, flag, data_df, vali_points_percentage):
    Data = data_dict['custom']
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(
        df=data_df,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        vali_points_percentage=vali_points_percentage
    )

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
