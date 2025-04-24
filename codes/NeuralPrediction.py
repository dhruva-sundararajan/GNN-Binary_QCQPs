# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch_geometric
import dhg
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import pprint
import os
from sklearn.metrics import f1_score

# global variables
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_epoch_list = [1, 10, 50, 100]  # number of epochs to visualize

# %%
all_activations = {
    'relu': nn.ReLU(),
    'prelu': nn.PReLU(),
    'elu': nn.ELU(),
    'leakyrelu': nn.LeakyReLU(negative_slope=0.1),  
    'sigmoid': nn.Sigmoid(),
    'tanh': nn.Tanh(),
    'softmax': nn.Softmax(),
    'identity': nn.Identity()
}

def loss_fn(output, y_matrix):
    obj_vals = y_matrix[:, 0]             
    solutions = y_matrix[:, 1:]           
    weights = torch.softmax(-obj_vals, dim=0)
    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    loss = 0.0
    n_solns = solutions.shape[0]
    for i in range(n_solns):
        target = solutions[i]
        weight = weights[i]
        bce = criterion(output, target)
        loss += weight * bce
    return loss


class UniEGNNConv(nn.Module):
    """
    This convolution utilizes the hypergraph structure.
    Message passes from vertex to (hyper)edge, then back to vertex.
    """

    def __init__(
            self,
            args,
            in_channels,
            out_channels,
            bias,
            drop_rate
    ):
        super().__init__()
        self.args = args
        self.v_channels = in_channels
        self.e_channels = out_channels
        self.act = all_activations[args.activation]
        self.drop = nn.Dropout(drop_rate)
        self.theta_vertex = nn.Linear(
            self.v_channels, self.e_channels, bias=bias)
        self.theta_edge = nn.Linear(
            self.e_channels, self.v_channels, bias=bias)
        self.edge_merge = nn.Linear(
            self.e_channels * 2, self.e_channels, bias=bias)
        self.vertex_merge = nn.Linear(
            self.v_channels * 2, self.v_channels, bias=bias)
        self.first_aggregate = args.first_aggregate
        self.second_aggregate = args.second_aggregate
        self.fix_edge = args.fix_edge
        self.layer_norm_v = nn.LayerNorm(self.v_channels)
        self.layer_norm_e = nn.LayerNorm(self.e_channels)

    def forward(self, X, Y, graph):
        X_0 = X
        Y_0 = Y
        X = self.theta_vertex(X)

        if isinstance(graph, dhg.BiGraph):
            Y = self.edge_merge(
                torch.cat([Y_0, graph.u2v(X, aggr=self.first_aggregate)], dim=-1))
            Y = self.theta_edge(Y)
            X = self.vertex_merge(
                torch.cat([X_0, graph.v2u(Y, aggr=self.second_aggregate)], dim=-1))

        elif isinstance(graph, dhg.Hypergraph):
            Y = self.edge_merge(
                torch.cat([Y_0, graph.v2e_aggregation(X, aggr=self.first_aggregate)], dim=-1))
            Y = self.theta_edge(Y)
            X = self.vertex_merge(
                torch.cat([X_0, graph.e2v_aggregation(Y, aggr=self.second_aggregate)], dim=-1))
        else:
            raise TypeError(
                "graph should be either dgg.BiGraph or dhg.Hypergraph.")

        X = self.layer_norm_v(X)
        Y = self.layer_norm_e(Y)
        X = self.drop(self.act(X))
        Y = self.drop(self.act(Y))
        if self.fix_edge:
            return X, Y_0
        else:
            return X, Y


all_convs = {
    'UniEGNN': UniEGNNConv,
}

class HyperGraphModel(nn.Module):

    def __init__(self, args):
        super().__init__()

        Conv = all_convs[args.model]

        self.convs = nn.ModuleList(
            [Conv(args, args.nhid, args.nhid, args.bias, args.drop_rate)
             for _ in range(args.nlayer)]
        )

        self.args = args
        self.act = all_activations[args.activation]
        self.vertex_input = nn.Sequential(
            nn.LayerNorm(args.nfeat),
            nn.Linear(args.nfeat, args.nhid, bias=args.bias),
            self.act,
            nn.Dropout(args.drop_rate),
            nn.Linear(args.nhid, args.nhid, bias=args.bias)
        )

        self.edge_input = nn.Sequential(
            nn.LayerNorm(args.nedge),
            nn.Linear(args.nedge, args.nhid, bias=args.bias),
            self.act,
            nn.Dropout(args.drop_rate),
            nn.Linear(args.nhid, args.nhid, bias=args.bias)
        )

        self.vertex_out = nn.Sequential(
            nn.LayerNorm(args.nhid),
            nn.Linear(args.nhid, args.nhid, bias=args.bias),
            self.act,
            nn.Dropout(args.drop_rate),
            nn.Linear(args.nhid, args.nout, bias=args.bias)
        )

    def forward(self, G):
        X = self.vertex_input(G.X)
        Y = self.edge_input(G.Y)
        for i, conv in enumerate(self.convs):
            X, Y = conv(X, Y, G)
        X = self.act(X)
        X = self.vertex_out(X)
        X = torch.sigmoid(X)
        return X.squeeze()


all_models = {
    'UniEGNN': HyperGraphModel
}


class HGData(dhg.Hypergraph):
    def __init__(self,
                 var_features: np.ndarray,
                 one_features: np.ndarray,
                 sqr_features: np.ndarray,
                 constr_features: np.ndarray,
                 obj_features: np.ndarray,
                 edge_features: np.ndarray,
                 edges: np.ndarray,
                 sol: np.ndarray
                 ):
        self.edges = torch.LongTensor(edges).T
        self.X = np.concatenate([var_features, one_features,
                                 sqr_features, constr_features, obj_features], axis=0)
        self.X = torch.FloatTensor(self.X)
        self.Y = torch.FloatTensor(edge_features).reshape(-1, 1)
        self.opt_sol = torch.FloatTensor(sol)
        super().__init__(num_v=self.X.shape[0], e_list=edges.tolist())

    def to(self, device: torch.cuda.device):
        super().to(device)
        self.edges = self.edges.to(device)
        self.X = self.X.to(device)
        self.Y = self.Y.to(device)
        self.opt_sol = self.opt_sol.to(device)
        return self


class HGDataset(torch.utils.data.Dataset):
    def __init__(self, sample_files):
        super().__init__()
        self.sample_files = sample_files

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, idx):
        name, graph = pickle.load(open(self.sample_files[idx], "rb"))
        return name, HGData(*graph)

all_data = {
    "HG": HGData,
}

all_datasets = {
    "HG": HGDataset,
}

def collate_fn(batch):
    [batch] = batch
    return batch


def train(args):
    difficulty = args.difficulty
    model_name = args.model
    nlayer = args.nlayer
    device = args.device
    nepoch = args.nepoch
    batch_size = args.batch_size
    out_dir = f'{_root}/runs/train/{args.encoding}/{model_name}/{difficulty}_layer{nlayer}_nepoch{nepoch}_lr{args.lr}_wd{args.wd}'
    os.makedirs(out_dir, exist_ok=True)
    data_dir = os.path.join(
        _root, 'data', 'train', args.encoding, f'train', difficulty)
    model_save_path = os.path.join(out_dir, 'trained_model.pkl')

    if os.path.exists(model_save_path):
        print(f"Model {model_save_path} exists.\nOverwrite? (y/n)")
        if input() not in ['Y', 'y']:
            print("Skip training.")
            return
        else:
            print("Overwriting trained model.")
            for f in os.listdir(out_dir):
                if os.path.isfile(os.path.join(out_dir, f)):
                    os.remove(os.path.join(out_dir, f))
    args_dict = vars(args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = all_models[args.model](args).to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.wd)

    print("Loading data...")

    sample_files = [os.path.join(data_dir, file)
                    for file in os.listdir(data_dir)]

    dataset = all_datasets[args.encoding](sample_files)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    data_loader = DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    print(f"Start training {model_name} on {difficulty} problems with {nlayer} layers.")

    losses = []
    early_stop_count = 0
    best_val_loss = float('inf')

    for epoch in range(nepoch):

        model.train()
        train_losses = []
        accumulated_loss = torch.tensor(0.0, device=device)
        count = 0

        for name, G in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):

            G = G.to(device)
            output = model(G)
            output = output[:(G.opt_sol.size()[1] - 1)]
            binary_output = (output > 0).long()
            loss = loss_fn(output, G.opt_sol)
            accumulated_loss += loss
            count += 1

            if count % batch_size == 0:
                optimizer.zero_grad()
                (accumulated_loss / batch_size).backward()
                optimizer.step()
                train_losses.append(accumulated_loss.item() / batch_size)
                accumulated_loss = torch.tensor(0.0, device=device)

        if count % batch_size != 0:
            optimizer.zero_grad()
            (accumulated_loss / (count % batch_size)).backward()
            optimizer.step()
            train_losses.append(accumulated_loss.item() / (count % batch_size))

        mean_train_loss = np.mean(train_losses, axis=0)
        print(f"Epoch {epoch+1} training: loss: {mean_train_loss:.5f}")

        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for _, G in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
                G = G.to(device)
                output = model(G)
                output = output[:(G.opt_sol.size()[1] - 1)]
                loss = loss_fn(output, G.opt_sol)
                val_losses.append(loss.item())

            mean_val_loss = np.mean(val_losses, axis=0)
            print(f"Epoch {epoch+1} validation: loss: {mean_val_loss:.5f}")

            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                early_stop_count = 0
                torch.save(model.state_dict(), model_save_path)
                print(f"Model {model_name} saved to {model_save_path}.")
            else:
                early_stop_count += 1
                if early_stop_count >= args.early_stop:
                    print(f"Early stop at epoch {epoch+1}.")
                    break
        losses.append(mean_val_loss)
            
        if epoch+1 in _epoch_list and args.output:
            with torch.no_grad():
                outputs = []
                for name, G in tqdm(data_loader, desc=f"Epoch {epoch+1} Output"):

                    G = G.to(device)

                    output = model(G)
                    output = output[:len(G.opt_sol)]

                    outputs.append((name, output.cpu().numpy()))

                pickle.dump(outputs, open(
                    f"{out_dir}/epoch_{epoch+1}_outputs.pkl", 'wb'))
                print(f"Epoch {epoch+1} outputs saved.")
    losses = np.array(losses)
    pickle.dump([losses], open(
        os.path.join(out_dir, "training_data.pkl"), "wb"))
    print(f"Training finished.\nBest validation loss: {min(losses):.5f}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--difficulty', '-d', type=str, required=True)
    parser.add_argument('--model', '-m', type=str, default='UniEGNN',
                        choices=['UniEGNN'])
    parser.add_argument('--encoding', '-e', type=str, default='HG',
                        choices=['HG'])
    parser.add_argument('--nlayer', type=int, default=6)
    parser.add_argument('--nout', type=int, default=1, choices=[1])
    parser.add_argument('--nhid', type=int, default=64)
    parser.add_argument('--drop_rate', type=float, default=0.1)
    parser.add_argument('--input_drop', type=float, default=0.0)
    parser.add_argument('--first_aggregate', type=str, default='sum',
                        choices=['sum', 'mean', 'softmax_then_sum'])
    parser.add_argument('--second_aggregate', type=str,
                        default='mean', choices=['sum', 'mean', 'softmax_then_sum'])
    parser.add_argument('--nobias', action='store_true', default=False)
    parser.add_argument('--fix_edge', '-f', action='store_true', default=False)
    parser.add_argument('--activation', type=str, default='leakyrelu')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--nepoch', type=int, default=100)
    parser.add_argument('--early_stop', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--vepoch', type=int, default=0,
                        help="0 for default epochs in one figure, -1 for default epochs in separate figures, integer for a specific epoch")
    parser.add_argument('--no_train', '-n', action='store_true',
                        default=False, help="train the model")
    parser.add_argument('--output', '-o', action='store_true',
                        default=False, help="output the neural outputs")
    args = parser.parse_args()
    args.bias = not args.nobias
    channels_dict = {
        'HG': (16, 1),
    }
    args.nfeat, args.nedge = channels_dict[args.encoding]

    return args


if __name__ == "__main__":
    args = parse_args()

    if not args.no_train:
        train(args=args)

# %%
