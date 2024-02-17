"""Parser for command line arguments."""
import argparse

def parse_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_train", type=int, help="Number of train samples", default=50_000)
    parser.add_argument("--n_test", type=int, help="Number of test samples", default=10_000)
    parser.add_argument("--model_size", type=int, help="Model size. Set to -1 to use the entire training dataset as model centers", default=20_000)
    parser.add_argument("--s_data", type=int, help="Number of Nystrom samples for Data Preconditioner", default=5_000)
    parser.add_argument("--q_data", type=int, help="Level of Data Preconditioner", default=100)
    parser.add_argument("--s_model", type=int, help="Number of Nystrom samples for Model Preconditioner", default=5_000)
    parser.add_argument("--q_model", type=int, help="Level of Model Preconditioner", default=100)
    parser.add_argument("--epochs", type=int, help="Number of training epochs", default=2)
    return parser.parse_args()    


