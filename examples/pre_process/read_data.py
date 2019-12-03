import pandas as pd


def read(data_name):
    if data_name == 'adult':
        path = '/home/xyan22/thesis/data/adult/'
        # path = '/Users/yanxinzhou/course/thesis/data/adult/'
        train = pd.read_csv(path + 'train.csv', index_col=False)
        test = pd.read_csv(path + 'test.csv', index_col=False)

        fea_train = ['Age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', \
                     'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', \
                     'native-country']
        fea_cat = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', \
                   'sex', 'native-country']

        train_x = train[fea_train]
        test_x = test[fea_train]
        train_y = train['salary']
        test_y = test['salary']

        for f in fea_cat:
            train_x[f] = train_x[f].astype('category')
            test_x[f] = test_x[f].astype('category')

        for f in fea_cat:
            mapping = dict(zip(list(train_x[f].cat.categories),
                               list(range(len(train_x[f].cat.categories)))))
            train_x = train_x.replace({f: mapping})
            test_x = test_x.replace({f: mapping})

        return train_x, train_y, test_x, test_y
