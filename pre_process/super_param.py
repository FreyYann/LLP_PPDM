import argparse


def parser_input():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--nmodel", dest='model_name', help="choose the model name", default='llp_lr',
                        action="store_true")
    parser.add_argument("-b", "--bag", dest='in_bag', help="whether in bag", default='True',
                        action="store_true")
    parser.add_argument("-m", "--bag_num", dest='bag_num', help="how many bags involved", default=300,
                        action="store_true")  # 300
    parser.add_argument("-I", "--bag_size", dest='bag_size', help="how many instances a bags involved", default=1000,
                        action="store_true")  # 100
    parser.add_argument("-t", "--eval_type", dest='eval_type', help="evaluation method", default='confusion_matrix',
                        action="store_true")
    parser.add_argument("-d", "--dimension", dest='dimension', help="demension of features", default=3136,
                        action="store_true")  # 14
    parser.add_argument("-s", "--source", dest='source', help="data source name", default='land',
                        action="store_true")

    parser.add_argument("-f", "--frequency", dest='frequency', help="frequency of train_x's row")
    parser.add_argument("-l", "--logger", dest='logger', help="logger")
    parser.add_argument("-bw", "--balance_weight", dest='balance_weight', help="balance_weight")
    parser.add_argument("-te", "--temp", dest='temp', help="temporary variant")

    return parser
