import pandas as pd
import torch
import numpy as np
import openml

from tabpfn.constants import DEFAULT_SEED
from tabpfn.datasets.preprocessing_utils import preprocessing

def get_openml_classification(did, max_samples, random_state=None, multiclass=True, shuffled=True, subsample_flag: bool = False):
    # Some datasets seem to have problems with `.pq` downloading, this forces
    # openml to use the `.arff` files instead
    # https://github.com/openml/openml-python/issues/1181#issuecomment-1321775563
    openml.datasets.functions._get_dataset_parquet = lambda x, **args: None

    dataset = openml.datasets.get_dataset(did)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute
    )

    X, y, categorical_indicator, num_high_cardinality, num_columns_missing, num_rows_missing, \
        num_categorical_columns, n_pseudo_categorical, original_n_samples, original_n_features = \
            preprocessing(X, y, categorical_indicator, categorical=False,
                        regression=False, transformation=None)
    # If subsampling, these column parameters will be modified
    return X, y, list(np.where(categorical_indicator)[0]), attribute_names

def load_openml_list(dids, random_state=None,
                     filter_for_nan=False,
                     num_feats=100,
                     min_samples = 100,
                     max_samples=400,
                     multiclass=True,
                     max_num_classes=10,
                     shuffled=True,
                     return_capped=False,
                     subsample_flag: bool = False):
    datasets = []
    openml_list = openml.datasets.list_datasets(dids)
    print(f'Number of datasets: {len(openml_list)}')

    datalist = pd.DataFrame.from_dict(openml_list, orient="index")
    if filter_for_nan:
        datalist = datalist[datalist['NumberOfInstancesWithMissingValues'] == 0]
        print(f'Number of datasets after Nan and feature number filtering: {len(datalist)}')

    # If we are going to be subsampling later, we need to ensure it ends up with a
    # different name
    if subsample_flag:
        datalist["name"] = datalist["name"].astype(str) + "-subsample-splits"

    for ds in datalist.index:
        # If we are going to subsample in the splits, these will all be capped
        modifications = {
            'samples_capped': subsample_flag,
            'classes_capped': subsample_flag,
            'feats_capped': subsample_flag,
        }
        entry = datalist.loc[ds]

        print('Loading', entry['name'], entry.did, '..')

        if entry['NumberOfClasses'] == 0.0:
            raise Exception("Regression not supported")
            #X, y, categorical_feats, attribute_names = get_openml_regression(int(entry.did), max_samples)
        else:
            X, y, categorical_feats, attribute_names = get_openml_classification(int(entry.did), max_samples,
                                                                                 multiclass=multiclass,
                                                                                 shuffled=shuffled,
                                                                                 random_state=random_state,
                                                                                 subsample_flag=subsample_flag)
        if X is None:
            continue


        print(f"Fetched {ds}")
        datasets += [[entry['name'], X, y, categorical_feats, attribute_names, modifications, int(entry.did)]]

    return datasets, datalist


# Classification
valid_dids_classification = [13, 59, 4, 15, 40710, 43, 1498]
test_dids_classification = [973, 1596, 40981, 1468, 40984, 40975, 41163, 41147, 1111, 41164, 1169, 1486, 41143, 1461, 41167, 40668, 41146, 41169, 41027, 23517, 41165, 41161, 41159, 41138, 1590, 41166, 1464, 41168, 41150, 1489, 41142, 3, 12, 31, 54, 1067]
valid_large_classification = [  943, 23512,    49,   838,  1131,   767,  1142,   748,  1112,
        1541,   384,   912,  1503,   796,    20,    30,   903,  4541,
         961,   805,  1000,  4135,  1442,   816,  1130,   906,  1511,
         184,   181,   137,  1452,  1481,   949,   449,    50,   913,
        1071,   831,   843,     9,   896,  1532,   311,    39,   451,
         463,   382,   778,   474,   737,  1162,  1538,   820,   188,
         452,  1156,    37,   957,   911,  1508,  1054,   745,  1220,
         763,   900,    25,   387,    38,   757,  1507,   396,  4153,
         806,   779,   746,  1037,   871,   717,  1480,  1010,  1016,
         981,  1547,  1002,  1126,  1459,   846,   837,  1042,   273,
        1524,   375,  1018,  1531,  1458,  6332,  1546,  1129,   679,
         389]

open_cc_dids = [11,
 14,
 15,
 16,
 18,
 22,
 23,
 29,
 31,
 37,
 50,
 54,
 188,
 458,
 469,
 1049,
 1050,
 1063,
 1068,
 1510,
 1494,
 1480,
 1462,
 1464,
 6332,
 23381,
 40966,
 40982,
 40994,
 40975]
# Filtered by N_samples < 2000, N feats < 100, N classes < 10

open_cc_valid_dids = [13,25,35,40,41,43,48,49,51,53,55,56,59,61,187,285,329,333,334,335,336,337,338,377,446,450,451,452,460,463,464,466,470,475,481,679,694,717,721,724,733,738,745,747,748,750,753,756,757,764,765,767,774,778,786,788,795,796,798,801,802,810,811,814,820,825,826,827,831,839,840,841,844,852,853,854,860,880,886,895,900,906,907,908,909,915,925,930,931,934,939,940,941,949,966,968,984,987,996,1048,1054,1071,1073,1100,1115,1412,1442,1443,1444,1446,1447,1448,1451,1453,1488,1490,1495,1498,1499,1506,1508,1511,1512,1520,1523,4153,23499,40496,40646,40663,40669,40680,40682,40686,40690,40693,40705,40706,40710,40711,40981,41430,41538,41919,41976,42172,42261,42544,42585,42638]

grinzstjan_categorical_regression = [44054,
 44055,
 44056,
 44057,
 44059,
 44061,
 44062,
 44063,
 44064,
 44065,
 44066,
 44068,
 44069]

grinzstjan_numerical_classification = [44089,
 44090,
 44091,
 44120,
 44121,
 44122,
 44123,
 44124,
 44125,
 44126,
 44127,
 44128,
 44129,
 44130,
 44131]

grinzstjan_categorical_classification = [44156, 44157, 44159, 44160, 44161, 44162, 44186]

automlbenchmark_ids = [181,
 1111,
 1596,
 1457,
 40981,
 40983,
 23517,
 1489,
 31,
 40982,
 41138,
 41163,
 41164,
 41143,
 1169,
 41167,
 41147,
 41158,
 1487,
 54,
 41144,
 41145,
 41156,
 41157,
 41168,
 4541,
 1515,
 188,
 1464,
 1494,
 1468,
 1049,
 23,
 40975,
 12,
 1067,
 40984,
 40670,
 3,
 40978,
 4134,
 40701,
 1475,
 4538,
 4534,
 41146,
 41142,
 40498,
 40900,
 40996,
 40668,
 4135,
 1486,
 41027,
 1461,
 1590,
 41169,
 41166,
 41165,
 40685,
 41159,
 41161,
 41150,
 41162,
 42733,
 42734,
 42732,
 42746,
 42742,  # Fails to load due to looking for a .pq file that has restricted access
 42769,
#43072, # Takes forever .. too large
]

benchmark_dids = dict(
    numerical = [
        44089, 44090, 44091, 44120, 44121, 44122, 44123, 44124, 44125, 44126, 44127, 44128, 44129, 44130, 44131
    ],
    categorical=[
        44156, 44157, 44159, 44160, 44161, 44162, 44186
    ]
)
too_easy_dids = dict(
    numerical = [
        44, 152, 153, 246, 251, 256, 257, 258, 267, 269, 351, 357, 720, 725, 734, 735, 737, 761, 803, 816, 819, 823, 833, 846, 847, 871, 976, 979, 1053, 1119, 1181, 1205, 1212, 1216, 1218, 1219, 1240, 1241, 1242, 1486, 1507, 1590, 4134, 23517, 41146, 41147, 41162, 42206, 42343, 42395, 42435, 42477, 42742, 42750, 43489, 60, 150, 159, 160, 180, #]
        182, 250, 252, 254, 261, 266, 271, 279, 554, 1110, 1113, 1183, 1185, 1209, 1214, 1222, 1226, 1351, 1352, 1353, 1354, 1355, 1356, 1357, 1358, 1359, 1360, 1361, 1362, 1363, 1364, 1365, 1366, 1368, 1393, 1394, 1395, 1476, 1477, 1478, 1503, 1526, 1596, 4541, 40685, 40923, 40996, 40997, 41000, 41002, 41039, 41163, 41164, 41166, 41168, 41169, 41671, 41972, 41982, 41986, 41988, 41989, 42468, 42746
         ],
    categorical=[
        4, 26, 154, 179, 274, 350, 720, 881, 923, 959, 981, 993, 1110, 1112, 1113, 1119, 1169, 1240, 1461, 1486, 1503, 1568, 1590, 4534, 4541, 40517, 40672, 40997, 40998, 41000, 41002, 41003, 41006, 41147, 41162, 41440, 41672, 42132, 42192, 42193, 42206, 42343, 42344, 42345, 42477, 42493, 42732, 42734, 42742, 42746, 42750, 43044, 43439, 43489, 43607, 43890, 43892, 43898, 43903, 43904, 43920, 43922, 43923, 43938
        ]
    )

not_too_easy_dids = dict(
    numerical = [
        # 41081
        6, 1037, 1039, 1040, 1044, 1557, 1558, 40983, 28, 30, 43551, 1056, 32, 38, 41004, 1069, 40498, 40499, 4154, 1597, 43072, 41027, 1111, 40536, 1112, 1114, 1116, 1120, 41082, 41103, 151, 41138, 183, 41150, 41156, 41160, 41161, 41165, 42192, 42193, 722, 727, 728, 40666, 42733, 40701, 42757, 42758, 42759, 40713, 41228, 42252, 42256, 273, 42769, 42773, 42774, 42775, 293, 300, 821, 310, 350, 354, 375, 390, 399, 42397, 1459, 1461, 1475, 40900, 40910, 1489, 977, 980, 41434, 41946, 993, 1000, 1002, 1018, 1019, 1021
    ]
)