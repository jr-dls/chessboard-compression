# import torch
# import torch.nn as nn
# import torch.optim as optim

# class SimpleNN(nn.Module):
#     def __init__(self):
#         super(SimpleNN, self).__init__()
#         self.fc1 = nn.Linear(2, 4)  
#         self.fc2 = nn.Linear(4, 1)  

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))  
#         x = self.fc2(x)               
#         return x

# X_train = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]) 
# y_train = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

# # Instantiate the Model, Define Loss Function and Optimizer
# model = SimpleNN()  
# criterion = nn.MSELoss()  
# optimizer = optim.SGD(model.parameters(), lr=0.1)

# for epoch in range(100):  
#     model.train() 

#     # Forward pass
#     outputs = model(X_train)
#     loss = criterion(outputs, y_train)  
    
#     # Backward pass and optimize
#     optimizer.zero_grad()  
#     loss.backward()  
#     optimizer.step()  

#     if (epoch + 1) % 10 == 0:  
#         print(f'Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}')

# model.eval()  
# with torch.no_grad(): 
#     test_data = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
#     predictions = model(test_data) 
#     print(f'Predictions:\n{predictions}')


import os
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from multiprocessing import freeze_support


class ChessNet(nn.Module):
    def __init__(self, input_dim=1560, hidden_dim=1040, num_pieces=12, max_count=11, num_squares=64):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_pieces = num_pieces
        self.max_count = max_count
        self.num_squares = num_squares

        # Capa oculta
        self.fc1 = nn.Linear(input_dim, hidden_dim)

        # Head 1: número de piezas (12 × 11)
        self.fc_num_pieces = nn.Linear(hidden_dim, num_pieces * max_count)

        # Head 2: existencia de pieza (12 × 64)
        self.fc_existence = nn.Linear(hidden_dim, num_pieces * num_squares)

    def forward(self, x):
        h = F.relu(self.fc1(x))

        # logits (se procesan después en la loss)
        logits_num = self.fc_num_pieces(h)       # (batch, 132)
        logits_existence = self.fc_existence(h)  # (batch, 768)

        return logits_num, logits_existence


class ChessDataset(IterableDataset):
    """
    IterableDataset que carga archivos .npz de a uno y produce ejemplos individuales.
    Cada ejemplo corresponde a un training case de numero_piezas o existencia_pieza.
    """
    def __init__(self, data_dir, shuffle_files=True, task_filter=None):
        """
        Args:
            data_dir (str): Directorio con archivos .npz.
            shuffle_files (bool): Si barajar la lista de archivos al inicio.
            task_filter (str|None): Si 'numero_piezas' o 'existencia_pieza', filtra el tipo de ejemplo.
        """
        super().__init__()
        self.data_dir = data_dir
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".npz")]
        if shuffle_files:
            np.random.shuffle(self.files)
        self.task_filter = task_filter

    def parse_file(self, path):
        """Itera sobre un archivo .npz y genera ejemplos individuales"""
        data = np.load(path)

        # Numero de piezas
        if self.task_filter in [None, "numero_piezas"]:
            inputs_num = data["inputs_num"]       # (N1, 1560)
            piece_indices_num = data["piece_indices_num"]  # (N1,)
            outputs_num = data["outputs_num"]     # (N1, 11)

            for x, idx, y in zip(inputs_num, piece_indices_num, outputs_num):
                yield {
                    "task": "numero_piezas",
                    "input": torch.from_numpy(x).float(),
                    "piece_index": torch.tensor(idx, dtype=torch.long),
                    "target": torch.tensor(np.argmax(y), dtype=torch.long)  # crossentropy target
                }

        # Existencia de pieza
        if self.task_filter in [None, "existencia_pieza"]:
            inputs_ex = data["inputs_ex"]         # (N2, 1560)
            piece_indices_ex = data["piece_indices_ex"]  # (N2,)
            squares_ex = data["squares_ex"]       # (N2,)
            labels_ex = data["labels_ex"]         # (N2,)

            for x, idx, sq, lbl in zip(inputs_ex, piece_indices_ex, squares_ex, labels_ex):
                yield {
                    "task": "existencia_pieza",
                    "input": torch.from_numpy(x).float(),
                    "piece_index": torch.tensor(idx, dtype=torch.long),
                    "square": torch.tensor(sq, dtype=torch.long),
                    "target": torch.tensor(lbl, dtype=torch.float32)  # BCE target
                }

    def __iter__(self):
        for path in self.files:
            yield from self.parse_file(path)


def chess_collate_fn(batch):
    """
    batch: lista de diccionarios, cada uno puede ser de tipo 'numero_piezas' o 'existencia_pieza'.
    Retorna un diccionario con tensores agrupados por tipo.
    """

    num_inputs, num_piece_indices, num_targets = [], [], []
    ex_inputs, ex_piece_indices, ex_squares, ex_targets = [], [], [], []

    for item in batch:
        if item["task"] == "numero_piezas":
            num_inputs.append(item["input"])
            num_piece_indices.append(item["piece_index"])
            num_targets.append(item["target"])
        else:  # existencia_pieza
            ex_inputs.append(item["input"])
            ex_piece_indices.append(item["piece_index"])
            ex_squares.append(item["square"])
            ex_targets.append(item["target"])

    result = {}
    if num_inputs:
        result["numero_piezas"] = {
            "inputs": torch.stack(num_inputs),
            "piece_indices": torch.tensor(num_piece_indices, dtype=torch.long),
            "targets": torch.tensor(num_targets, dtype=torch.long),
        }
    if ex_inputs:
        result["existencia_pieza"] = {
            "inputs": torch.stack(ex_inputs),
            "piece_indices": torch.tensor(ex_piece_indices, dtype=torch.long),
            "squares": torch.tensor(ex_squares, dtype=torch.long),
            "targets": torch.stack(ex_targets),  # ya son float32
        }
    return result


def train(model, dataloader, device, epochs=5, lr=1e-3):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    ce_loss = nn.CrossEntropyLoss()
    bce_loss = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss, total_batches = 0.0, 0
        total_num, total_ex = 0, 0

        for batch in dataloader:
            optimizer.zero_grad()
            loss = 0.0

            # --- numero_piezas ---
            if "numero_piezas" in batch:
                data = batch["numero_piezas"]["inputs"].to(device)
                piece_idx = batch["numero_piezas"]["piece_indices"].to(device)
                targets = batch["numero_piezas"]["targets"].to(device)

                logits_num, _ = model(data)

                # Vectorizado: selecciona bloques [piece_idx*11 : piece_idx*11+11]
                batch_size = logits_num.size(0)
                arange_idx = torch.arange(batch_size, device=device)
                selected = logits_num.view(batch_size, -1, 11)[arange_idx, piece_idx]

                loss_num = ce_loss(selected, targets)
                loss += loss_num
                total_num += targets.size(0)

            # --- existencia_pieza ---
            if "existencia_pieza" in batch:
                data = batch["existencia_pieza"]["inputs"].to(device)
                piece_idx = batch["existencia_pieza"]["piece_indices"].to(device)
                squares = batch["existencia_pieza"]["squares"].to(device)
                targets = batch["existencia_pieza"]["targets"].to(device)

                _, logits_ex = model(data)

                # Seleccionar valores relevantes [piece_index*64 + square]
                idx = piece_idx * 64 + squares
                selected = logits_ex.gather(1, idx.view(-1, 1)).squeeze(1)

                loss_ex = bce_loss(selected, targets)
                loss += loss_ex
                total_ex += targets.size(0)

            # --- backward ---
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

            if (total_batches % 1000 == 0):
                print(f"Se ha operado en {total_batches} batches y {64*total_batches} tuplas.")

        avg_loss = total_loss / max(1, total_batches)
        print(f"Epoch {epoch}: loss={avg_loss:.4f}, num_samples={total_num}, ex_samples={total_ex}")


if __name__ == "__main__":
    freeze_support()  # Esto es opcional, pero recomendable en Windows

    dataset = ChessDataset("C:/Users/IN_CAP02/Documents/dataset")  
    loader = DataLoader(dataset, batch_size=64, collate_fn=chess_collate_fn, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessNet()

    # train(model, loader, device, epochs=10, lr=1e-3)
    train(model, loader, device, epochs=1, lr=1e-3)

    # Guardar modelo
    torch.save(model.state_dict(), "chessnet.pth")
