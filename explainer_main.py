"""
    explainer_main.py
    Main user interface for the explainer module.
"""

import argparse
import os
import torch
from tensorboardX import SummaryWriter

from graphLambda import Net  # Import your model
from torch_geometric.data import DataLoader
import utils.io_utils as io_utils
import utils.parser_utils as parser_utils
from explainer import explain


def arg_parse():
    parser = argparse.ArgumentParser(description="GNN Explainer arguments.")
    parser.add_argument("--dataset", dest="dataset", help="Input dataset path.")
    parser.add_argument("--model-path", dest="model_path", help="Path to the trained model.")
    parser.add_argument("--logdir", dest="logdir", default="log", help="Tensorboard log directory.")
    parser.add_argument("--ckptdir", dest="ckptdir", help="Model checkpoint directory.")
    parser.add_argument("--cuda", dest="cuda", default="0", help="CUDA device ID.")
    parser.add_argument("--epochs", dest="num_epochs", type=int, default=200, help="Number of epochs for explainer.")
    parser.add_argument("--graph-idx", dest="graph_idx", type=int, default=-1, help="Graph to explain.")
    parser.add_argument("--explain-node", dest="explain_node", type=int, help="Node to explain.")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--mask-act", dest="mask_act", default="sigmoid", help="Mask activation function.")
    parser.add_argument("--graph-mode", dest="graph_mode", action="store_true", help="Run explainer in graph mode.")
    parser.add_argument("--gpu", dest="gpu", action="store_true", help="Use GPU if available.")
    return parser.parse_args()


def main():
    args = arg_parse()

    # Set device
    device = torch.device(f"cuda:{args.cuda}" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load your trained model
    model = Net().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print("Model loaded successfully.")

    # Load dataset
    dataset = torch.load(args.dataset)  # Assuming the dataset is saved as a PyTorch object
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Configure Tensorboard logging
    writer = SummaryWriter(args.logdir)

    # Initialize the explainer
    from torch_geometric.nn import GNNExplainer
    explainer = GNNExplainer(model, epochs=args.num_epochs, return_type='regression', mask_act=args.mask_act)

    # Run the explainer
    if args.explain_node is not None:
        print(f"Explaining node {args.explain_node}...")
        data = dataset[0]  # Assuming node explanations focus on the first graph in the dataset
        explanation = explainer.explain_node(args.explain_node, data.x.to(device), data.edge_index.to(device))
        print("Node explanation completed.")
    elif args.graph_mode and args.graph_idx >= 0:
        print(f"Explaining graph {args.graph_idx}...")
        data = dataset[args.graph_idx]
        explanation = explainer.explain_graph(data.x.to(device), data.edge_index.to(device))
        print("Graph explanation completed.")
    else:
        print("Please specify a node or graph to explain.")

    # Save explanation results
    explanation_path = os.path.join(args.logdir, "explanation.pt")
    torch.save(explanation, explanation_path)
    print(f"Explanation saved to {explanation_path}")

    writer.close()


if __name__ == "__main__":
    main()
