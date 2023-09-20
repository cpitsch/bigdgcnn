# PM4Py
from pm4py.objects.log.obj import EventLog
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.util import xes_constants as xes
# from pm4py import fitness_token_based_replay
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness
# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, Linear
## Torch_Geometric
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv
from torch_geometric.nn.aggr import SortAggregation
from torch_geometric import seed_everything

# Misc
from sklearn.model_selection import train_test_split as sklearn_train_test_split
import networkx as nx
import numpy as np
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt

# Internal
from bigdgcnn.data_processing import discover_model_imf
from bigdgcnn.util import add_artificial_start_end_events
from bigdgcnn.datasets import BIG_Instancegraph_Dataset
from bigdgcnn.datasets import BIG_Instancegraph_Dataset_With_Attributes

class DGCNN(nn.Module):

    def __init__(self,
            dataset: Dataset,
            graph_conv_layer_sizes: List[int],
            sort_pool_k: int,
            sizes_1d_convolutions: List[int],
            dense_layer_sizes: List[int],
            dropout_rate: float,
            learning_rate: float,
            activities_index: List[str],
            use_cuda_if_available: bool = True
        ):
        """A Deep Graph Convolutional Neural Network (DGCNN) for next-activity prediction. Based on the paper "Exploiting Instance Graphs and Graph Neural Networks for Next Activity Prediction" by Chiorrini et al.

        Args:
            dataset (Dataset): The dataset to construct the model for.
            graph_conv_layer_sizes (List[int]): The output size for each graph convolution layer.
            sort_pool_k (int): The number of nodes to select in the SortPooling layer.
            sizes_1d_convolutions (List[int]): The output size for each 1D convolution layer.
            dense_layer_sizes (List[int]): The output size for each dense layer.
            dropout_rate (float): The dropout rate to use in the Dropout layer after the 1D convolution.
            learning_rate (float): The learning rate to use for the optimizer.
            activities_index (List[str]): The list of activities in the log. The same as used for one-hot encoding. Used to determine the number of input features.
            use_cuda_if_available (bool, optional): Whether to use CUDA if available. Defaults to True. If CUDA is not available, CPU will be used either way.
        """        



        super().__init__()

        if use_cuda_if_available and torch.cuda.is_available():
            self.device = torch.device("cuda:0") #TODO: Add support for multiple GPUs?
            print("Using GPU")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")

        self.graph_conv_layer_sizes = graph_conv_layer_sizes
        self.sort_pool_k = sort_pool_k
        self.sizes_1d_convolutions = sizes_1d_convolutions
        self.dense_layer_sizes = dense_layer_sizes
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate

        #TODO: Research kernel size in a bit more detail...
        # The problem was that if sort_pool_k is too small, the input of the first dense layer has negative dimension
        self.KERNEL_SIZE = min(sort_pool_k, len(graph_conv_layer_sizes))

        self.num_features = dataset.num_node_features
        self.num_output_features = len(activities_index) # One-hot encoding of the activity

        # Graph Convolutions
        self.conv1 = SAGEConv(self.num_features, graph_conv_layer_sizes[0])
        self.convs = torch.nn.ModuleList()
        for in_size, out_size in zip(graph_conv_layer_sizes, graph_conv_layer_sizes[1:]):
            self.convs.append(SAGEConv(in_size, out_size))

        # In forward, there is some tensor magic between these two "layers"

        # 1D Convolution
        self.conv1d = Conv1d(graph_conv_layer_sizes[-1], sizes_1d_convolutions[0], self.KERNEL_SIZE)
        self.conv1ds = torch.nn.ModuleList()
        for in_size, out_size in zip(sizes_1d_convolutions, sizes_1d_convolutions[1:]):
            self.conv1ds.append(Conv1d(in_size, out_size, self.KERNEL_SIZE))

        # Dropout done in `forward`

        # Dense Layers
        # Input size from the source code
        #TODO: If len(dense_layer_sizes) == 0 this would fail
        self.linear = torch.nn.Linear(dense_layer_sizes[0] * (self.sort_pool_k - self.KERNEL_SIZE + 1), dense_layer_sizes[0])
        self.linears = torch.nn.ModuleList()
        for in_size, out_size in zip(dense_layer_sizes, dense_layer_sizes[1:]):
            self.linears.append(Linear(in_size, out_size))
        self.linear_output = Linear(dense_layer_sizes[-1], self.num_output_features) # Final layer for output

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.conv1d.reset_parameters()
        for conv1d in self.conv1ds:
            conv1d.reset_parameters()
        self.linear.reset_parameters()
        for linear in self.linears:
            linear.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # Graph Convolution
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        # Sort-Pooling
        module = SortAggregation(k=self.sort_pool_k)
        x = module(x, batch)

        # Weird tensor magic from the source code
        x = x.view(len(x), self.sort_pool_k, -1).permute(0, 2, 1) # modification of the structure of the vector to be able to pass it to the conv1d layer (they must have n°nodes=k) (translated from their source code)

        # 1D Convolutions
        x = F.relu(self.conv1d(x))
        for conv1d in self.conv1ds:
            x = F.relu(conv1d(x))

        # Reshape the tensor to be able to pass it to the dense layers (flatten ?)
        x = x.view(len(x), -1)

        # Dropout
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # Dense Layers
        x = F.relu(self.linear(x))
        for linear in self.linears:  
            x = F.relu(linear(x))
        x = F.relu(self.linear_output(x))
        return x # No activation function, as pytorch already applies softmax in cross-entropy loss

    def __repr__(self):
        params = {
            "Graph Conv. Layer Sizes": self.graph_conv_layer_sizes,
            "Sort Pool K": self.sort_pool_k,
            "1D Conv. Sizes": self.sizes_1d_convolutions,
            "Dense Layer Sizes": self.dense_layer_sizes,
            "Dropout Rate": self.dropout_rate,
            "Learning Rate": self.learning_rate,
            "Distinct Activities": self.num_output_features,
        }
        return self.__class__.__name__ + " Model: " + str(params)
        

class BIG_DGCNN():
    
    def __init__(self,
                 layer_sizes: List[int],
                 sort_pooling_k: int,
                 sizes_1d_convolutions: List[int] = [32],
                 dense_layer_sizes: List[int] = [32],
                 dropout_rate: float = 0.1,
                 learning_rate: float = 1e-3,
                 batch_size: int = 32,
                 epochs: int = 100,
                 seed: Optional[int] = None,
                use_cuda_if_available: bool = True
                ):
        """A Big-DGCNN model for next-activity prediction.

        Args:
            layer_sizes (List[int]): The output sizes of each graph convolution layer. Also determines the number of graph convolutions to perform. In the paper this count is varied in {2,3,5,7}, with each layer having size 32.
            sort_pooling_k (int): The number of nodes to select in the SortPooling layer. In the paper this is varied in {3, 5, 30}.
            sizes_1d_convolutions (List[int], optional): The sizes of the 1D convolutions. Defaults to [32], i.e., *one* 1D convolution with size 32, as used in the paper.
            dense_layer_sizes (List[int], optional): The sizes of the dense layers. Defaults to [32], i.e., *one* dense layer with size 32, as used in the paper.
            dropout_rate (float, optional): The dropout rate to use in the Dropout layer after the 1D convolution. Defaults to 0.1. In the paper this is varied in {0.1, 0.2, 0.5}
            learning_rate (float, optional): The learning rate to use for the optimizer. Defaults to 10^(-3). In the paper this is varied in {10^(-2), 10^(-3), 10^(-4)}
            batch_size (int, optional): The batch size to use for training. Defaults to 32. In the paper this is varied in {16, 32, 64, 128}
            epochs (int, optional): The number of epochs to train for. Defaults to 100, the value used in the paper.
            seed (int, optional): The seed to use for reproducibility. Defaults to None (No seed). Currently does not work as intended.
            use_cuda_if_available (bool, optional): Whether to use CUDA if available. Defaults to True.
        """
        if seed is not None:
            seed_everything(seed)
        self.seed = seed

        self.layer_sizes = layer_sizes
        self.sort_pooling_k = sort_pooling_k
        self.sizes_1d_convolutions = sizes_1d_convolutions
        self.dense_layer_sizes = dense_layer_sizes
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.use_cuda_if_available = use_cuda_if_available

    def train(self, 
                  log: EventLog,
                  logname: str,
                  process_model: Optional[Tuple[PetriNet, Marking, Marking]]=None,
                  imf_noise_thresh: Optional[float] =None,
                  train_test_split: float=0.67,
                  train_validation_split: float=0.8,
                  activityName_key:str=xes.DEFAULT_NAME_KEY,
                  torch_load_path:  Optional[str] =None,
                  case_level_attributes: Optional[List[str]] =None,
                  event_level_attributes: Optional[List[str]] =None,
                  force_reprocess_dataset: bool=False
                ):
        """Train the Deep Graph Convolutional Neural Network (DGCNN) for next activity prediction on a log and a process model.

        Args:
            log (EventLog): The event log to use. Artificial start/end events will be added internally.
            logname (str): The name of the event log/dataset. Used for caching/loading the dataset.
            process_model (Tuple[PetriNet, Marking, Marking], optional): The Process Model (A Petri Next with initial and final marking). If not provided, one will be mined using the IMf Algorithm. Note that if you provide a model, it should contain the artificial start and end events already.
            imf_noise_thresh (float, optional): The noise threshold to be used for the Inductive Miner Infrequent algorithm if it is used (process_model is None). Defaults to None. If no process model is provided, and no imf_noise_thresh is provided, the lowest noise threshold (rounded to 10) generating a model with >=90% fitness will be used.
            activityName_key (str, optional): The key used for the activity label in the event log. Defaults to xes.DEFAULT_NAME_KEY (concept:name).
            torch_load_path (str, optional): The path to a saved model to load. If provided, any training will be skipped and the model will be loaded from the given path. Defaults to None.
            case_level_attributes (List[str], optional): The list of case-level attributes to use for training. Defaults to None, which means empty list.
            event_level_attributes (List[str], optional): The list of event-level attributes to use for training. Defaults to None, which means empty list.
        """
        case_level_attributes = case_level_attributes if case_level_attributes is not None else []
        event_level_attributes = event_level_attributes if event_level_attributes is not None else []

        log = add_artificial_start_end_events(log, activityName_key=activityName_key)
        if process_model is None:
            if imf_noise_thresh is not None: # Specific noise threshold given --> Use This one
                process_model = discover_model_imf(log, imf_noise_thresh)
            else: # No specific noise threshold given --> find a decent one
                # Choose the lowest noise threshold that still generates a model with >=90% fitness
                noise_thresh = 1.0
                condition = True # To emulate a do-while loop
                while condition:
                    process_model = discover_model_imf(log, noise_thresh)
                    noise_thresh -= 0.1
                    # condition = fitness_token_based_replay(log, *process_model)['percentage_of_fitting_traces'] < 0.9 and noise_thresh >= 0  
                    fitness_results = replay_fitness.apply(
                        log,
                        *process_model,
                        variant=replay_fitness.Variants.TOKEN_BASED,
                        parameters={'show_progress_bar': False}
                    )
                    condition = fitness_results['percentage_of_fitting_traces'] < 0.9 and noise_thresh >= 0  
                print(f"Discovered model using IMf with noise threshold {noise_thresh+0.1}")
        self.process_model = process_model
        # self.dataset = BIG_Instancegraph_Dataset(log, logname + f"_noisethresh_{int((noise_thresh+0.1)*100)}", process_model=process_model) # Caution: If the dataset has been created before using the same logname, it will be loaded from saved files, so changes in log, process_model will have no effect. To circumvent this, set force_reprocess=True
        self.dataset = BIG_Instancegraph_Dataset_With_Attributes(
            log,
            logname + f"_noisethresh_{int((noise_thresh+0.1)*100)}",
            process_model=process_model,
            case_level_attributes=case_level_attributes,
            event_level_attributes=event_level_attributes,
            force_reprocess=force_reprocess_dataset
        ) # Caution: If the dataset has been created before using the same logname, it will be loaded from saved files, so changes in log, process_model will have no effect. To circumvent this, set force_reprocess=True

        self.activities_index = self.dataset.activities_index


        # shuffle=False for a chronological split like in the paper
        ## When using a seed, the training is not reproducible unless I set the random_state here. Which doesn't make sense because according
        ## To the documentation, random_state controls the shuffling (which is turned off)...
        self.train_split, self.test_split = sklearn_train_test_split(self.dataset, train_size=train_test_split, shuffle=False, random_state=self.seed)
        self.train_split, self.validation_split = sklearn_train_test_split(self.train, train_size=train_validation_split, shuffle=False, random_state=self.seed)

        self.train_data = DataLoader(self.train_split, batch_size=self.batch_size, shuffle=True)
        self.validation_data = DataLoader(self.validation_split, batch_size=self.batch_size, shuffle=True)
        self.test_data = DataLoader(self.test_split, batch_size=self.batch_size, shuffle=True)


        if torch_load_path is not None:
            self.model = torch.load(torch_load_path)
            print(f"Model loaded from {torch_load_path}")
        else:
            self.model = DGCNN(
                dataset=self.dataset,
                graph_conv_layer_sizes=self.layer_sizes,
                sort_pool_k=self.sort_pooling_k,
                sizes_1d_convolutions=self.sizes_1d_convolutions,
                dense_layer_sizes=self.dense_layer_sizes,
                dropout_rate=self.dropout_rate,
                learning_rate=self.learning_rate,
                activities_index=self.activities_index,
                use_cuda_if_available=self.use_cuda_if_available
            )
            self.model = self.model.to(self.model.device)

            self._training_loop()
            print(f"Training Completed")

        self.test_accuracy = self.evaluate(self.test_data)*100
        print(f"Accuracy on Test Set: {self.test_accuracy:.4f}%")

    def _training_loop(self):
        self.model.train(True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        # criterion = torch.nn.L1Loss()

        self.train_losses = []
        self.validation_losses = []

        self.train_accuracies = []
        self.validation_accuracies = []

        epoch_losses = []
        for epoch in range(self.epochs):
            self.model.train(True)
            train_loss = 0
            for b in self.train_data: # Trainings
                batch = b.to(self.model.device)
                optimizer.zero_grad(set_to_none=True)

                out = self.model(batch)
                label = batch.y.view(out.shape[0],-1) # out.shape[0] to get the size of the current batch. (Last batch can be smaller if batch_size does not divide the number of instances)

                loss = criterion(out, label)

                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            valid_loss = 0

            for b in self.validation_data: # Validation
                batch = b.to(self.model.device)
                out = self.model(batch)
                label = batch.y.view(out.shape[0],-1)
                loss = criterion(out, label)
                valid_loss += loss.item()
            this_epoch_losses = (train_loss/len(self.train_data), valid_loss/len(self.validation_data)) # Average loss over the batches
            epoch_losses.append(this_epoch_losses)

            valid_accuracy = self.evaluate(self.validation_data)
            print(f"Epoch {epoch+1} completed. Train. Loss: {this_epoch_losses[0]}, Valid. Loss: {this_epoch_losses[1]}; Valid. Accuracy: {valid_accuracy*100:.4f}%")

            self.train_losses.append(this_epoch_losses[0])
            self.validation_losses.append(this_epoch_losses[1])

            self.train_accuracies.append(self.evaluate(self.train_data)) # model set to eval in here. No problem since we set train at the top
            self.validation_accuracies.append(valid_accuracy)


    def evaluate(self, test_dataset: BIG_Instancegraph_Dataset) -> float:
        """Calculate the accuracy of the model on a test dataset.

        Args:
            test_dataset (BIG_Instancegraph_Dataset): The dataset to evaluate on.

        Returns:
            float: The computed accuracy
        """        

        num_correct = 0
        total = 0
        self.model.eval()

        with torch.no_grad():
            for b in test_dataset:
                batch = b.to(self.model.device)

                out = self.model(batch)
                label = batch.y.view(out.shape)

                predictions = torch.argmax(out, dim=1)
                ground_truth = torch.argmax(label, dim=1).to(self.model.device)

                total += len(predictions)
                num_correct += torch.sum(predictions == ground_truth).item()
        return num_correct/total


    def evaluate_for_activity_set(self, test_dataset: BIG_Instancegraph_Dataset, activities: List[str], return_total: bool=False) -> float | Tuple[float, int]:
        """Calculate the accuracy of the model on a test dataset, but only for a given set of activities. 
        I.e., consider only the datapoints which are labelled with one of the given activities.

        Args:
            test_dataset (BIG_Instancegraph_Dataset): The dataset to evaluate on.
            activities (List[str]): The list of activities to consider.
            return_total (bool, optional): Whether to return the total number of datapoints labelled with one of the considered activities. Defaults to False.

        Returns:
            float: The computed accuracy
        """
        num_correct = 0
        total = 0
        self.model.eval()

        with torch.no_grad():
            for b in test_dataset:
                batch = b.to(self.model.device)

                out = self.model(batch)
                label = batch.y.view(out.shape)

                predictions = torch.argmax(out, dim=1)
                ground_truth = torch.argmax(label, dim=1).to(self.model.device)


                considered = [
                    (prediction, ground_truth)
                    for prediction, ground_truth in zip(predictions, ground_truth)
                    if self.activities_index[ground_truth.item()] in activities
                ]
                total += len(considered)
                # num_correct += torch.sum(predictions == ground_truth).item()
                num_correct += len([x for x in considered if x[0] == x[1]])
        return num_correct/total if not return_total else (num_correct/total, total)

    def save(self, path: str):
        """Save the model using the torch.save function. Thus, the file will have the extension `.pt`.

        Args:
            path (str): The path to save the model to.
        """
        torch.save(self.model, path)

    def plot_training_history(self) -> plt.Figure:
        fig, ax = plt.subplots(2, 1, figsize=(8, 10))

        ax[1].plot(self.train_losses, label="Train")
        ax[1].plot(self.validation_losses, label="Validation")
        ax[1].set_ylabel("Loss", fontsize=14)

        ax[0].plot(self.train_accuracies, label="Train")
        ax[0].plot(self.validation_accuracies, label="Validation")
        ax[0].set_ylabel("Accuracy", fontsize=14)

        ax[1].set_xlabel("Epoch", fontsize=14)
        ax[0].legend(loc="upper right")

        return fig