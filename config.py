import os

cls = ["pizza", "farm", "forest_road", "butte"]
dic = {
    "pizza": (4481, 497),  # update this to match exatly
    "farm": (4484, 498),
    "forest_road": (4472, 496)
}


class config:
    c = cls[2]
    train_dir = f"/Users/naveen/Documents/ML local Data/{c}"
    test_dir = f"/Users/naveen/Documents/ML local Data/testdata/{c}"
    embedding_path = f"/Users/naveen/PycharmProjects/ZhanghImpl1/encoder_decoder/embedding/{c}"
    train_size = dic[c][0]
    validation_size = dic[c][1]

    results_dir = "/Users/naveen/PycharmProjects/ZhanghImpl1/zhang_results"
    results_dir_emd = "/Users/naveen/PycharmProjects/ZhanghImpl1/encoder_decoder/results_ed"

    color_data_path = "/Users/naveen/PycharmProjects/ZhanghImpl1/color_data"
    prior_factor_file_path = os.path.join(color_data_path, "prior_factor.npy")
    H = 128
    W = 128
    batch_size = 100
    scale = 0.25
    upscale = int(1 / 0.25)
    h = int(H * scale)  # model_out size
    w = int(W * scale)
    neighbours_soft_encoding = 5
    Q = 313

    '''
    model related configs
    '''
    model_min_loss_out = f"/Users/naveen/PycharmProjects/ZhanghImpl1/{c}model.h5"
    model_min_val_loss_out = f"/Users/naveen/PycharmProjects/ZhanghImpl1/{c}model_val.h5"

    model_min_loss_out_emb = f"/Users/naveen/PycharmProjects/ZhanghImpl1/encoder_decoder/{c}model.h5"
    model_min_val_loss_out_emd = f"/Users/naveen/PycharmProjects/ZhanghImpl1/encoder_decoder/{c}model_val.h5"
    epochs = 40
    T = 0.38  # annealed mean temperature
