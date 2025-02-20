import torch
import torch.optim as optim
import numpy as np
import os
import time
import gc
import yaml
import pickle
from contextlib import redirect_stdout
from utils.utils import *
from utils.data_utils import get_dataloaders, get_dataset_dims
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from models.TimeGNN import TimeGNN

#relative path set-up
base_path = os.path.dirname(os.path.abspath(__file__))  # Huidige scriptmap

#load experiment configs
with open(os.path.join(base_path, 'Experiment_config.yaml'), 'r') as f:
    config = list(yaml.load_all(f, Loader=yaml.SafeLoader))[0]
    
def train(model):
    model.train()
    batch_losses = [] 

    for batch in train_loader:
        x_batch, y_batch = batch[0].float().to(device), batch[1].float().to(device)       
      
        y_hat = model(x_batch)      
      
        loss = model.loss(y_hat, y_batch)
        
        # Computes gradients
        loss.backward()

        # Updates parameters and zeroes gradients
        optimizer.step()
        optimizer.zero_grad()

        batch_losses.append(loss.item())
    
    return batch_losses 

def val(model):
    model.eval()
    batch_val_losses = []    

    for batch in val_loader:
        x_batch, y_batch = batch[0].float().to(device), batch[1].float().to(device)

        y_hat = model(x_batch)

        loss = model.loss(y_hat, y_batch)
        
        batch_val_losses.append(loss.item())
        
    return np.mean(batch_val_losses)

def test(test_loader, load_state = True, model_loc = "", return_graphs = False):
    if load_state:
        model.load_state_dict(torch.load(model_loc))
      
    with torch.no_grad():
        model.eval()        
        predictions = []
        values = []
        
        for batch in test_loader:
            x_batch, y_batch = batch[0].float().to(device), batch[1].float().to(device)


            if return_graphs: 
                y_hat, graph = model(x_batch, return_graphs)
                edge_list = dense_to_sparse(graph)[0]
                print(edge_list)
            else:
                y_hat = model(x_batch)
            
            y_hat = y_hat.cpu().detach().numpy()
            predictions.append(y_hat)
            values.append(y_batch.cpu().detach().numpy())            

    return predictions, values

#Recording args
model_type = "TimeGNN"
output_dir = os.path.join(os.getcwd(), "Time-GNN", "Experiments")
experiment_number = None #Leave as None to autogenerate a new folder. Change to write into a specific folder
save_dir, model_dir = create_directories(model_type, output_dir, experiment_number)
print(save_dir)

#Dataset args    
input_dim, output_dim = get_dataset_dims(config["dataset"],config["features"])

#Load data
train_loader, val_loader, test_loader, test_loader_one, scaler = get_dataloaders(config["dataset"], seq_len = config["seq_len"], 
                                                                                 horizon = config["horizon"], features = config["features"], 
                                                                                 cut = config["cut"])

#Model Args 
args = {
    "input_dim" : input_dim, 
    "output_dim" : output_dim,
    "seq_len" : config["seq_len"],  
    
    "hidden_dim" : 32,    
    "batch_size" : 16,    
    
    #Edge Learning Args
    "keep_self_loops" : False,
    "enforce_consecutive" : False,
    
    #GNN Args
    "block_size" : 3, 
    "aggregate" : "last",

}
loss = masked_mae
 
#Training args
train_losses = [] 
val_losses = []
val_epoch = [] 
metrics_last = {}
metrics_best = {}
results_last = []
results_best = [] 

#Save args
with open(save_dir + model_type + "_args.yaml", 'w') as f:
    yaml.dump(args, f, sort_keys=False, default_flow_style=False)
    
with open(save_dir + "Experiment_config.yaml", "w") as f:
    yaml.dump(config, f, sort_keys=False, default_flow_style=False)
    
#Training
did_nan = False 
n_epochs = config["n_epochs"]
for i in range(0, config["runs"]):
    print("Run " + str(i))
    best_val = float('inf')
    
    model = None
    gc.collect()
    torch.cuda.empty_cache()

    model = TimeGNN(loss, **args).to(device)
    optimizer = optim.Adam(model.parameters(), weight_decay = 1e-5)
    
    train_time = []    
    for epoch in range(config["n_epochs"]):
        t0 = time.time()
        batch_losses = train(model)
        t1 = time.time()
        
        train_losses.append(np.mean(batch_losses))
        train_time.append(t1-t0)

        if epoch % config["val_interval"] == 0:
            val_losses.append(val(model))
            val_epoch.append(epoch)
            print(
              f"[{epoch}/{n_epochs}] Training loss: {train_losses[-1]:.4f}\t Validation loss: {val_losses[-1]:.4f} \t Time: {t1-t0:.2f}"
          )
            if val_losses[-1] <= best_val and not np.isnan(val_losses[-1]):
                best_val = val_losses[-1]
                torch.save(model.state_dict(), model_dir + "best_run" + str(i) + ".pt")   
        else:
            print(f"[{epoch}/{n_epochs}] Training loss: {train_losses[-1]:.4f} \t Time: {t1-t0:.2f}")

    torch.save(model.state_dict(), model_dir + "last_run" + str(i) + ".pt")

    print("Last model this run: ")
    t0 = time.time()
    predictions, values = test(test_loader_one, load_state = False)
    t1 = time.time()
    inf_time = t1-t0
    metrics_last, df_results_last = metrics(predictions, values, metrics_best, scaler, test_loader_one.dataset.start, config["features"], train_time, inf_time)
    results_last.append(df_results_last)  

    print("")
    print("Best model this run: ")
    t0 = time.time()
    predictions, values = test(test_loader_one, load_state = True, model_loc = model_dir + "best_run" + str(i) + ".pt")
    t1 = time.time()    
    inf_time = t1-t0
    metrics_best, df_results_best = metrics(predictions, values, metrics_best, scaler, test_loader_one.dataset.start, config["features"], train_time, inf_time)

    results_best.append(df_results_best)
    print("")

print("DONE")
print(model_type)
print(config["dataset"])
print("horizon " + str(config["horizon"]))

print("\nBest Models Aggregate:")
print_metrics(metrics_best)
print("\nLast Models Aggregate:")
print_metrics(metrics_last)

with open(save_dir + 'out.txt', 'w') as f:
    with redirect_stdout(f):
        print(model_type)
        print(config["dataset"])
        print("horizon " + str(config["horizon"]))

        print("\nBest Models Aggregate:")
        print_metrics(metrics_best)
        print("\nLast Models Aggregate:")
        print_metrics(metrics_last)
        
with open(save_dir + model_type + "_best.pickle", 'wb') as handle:
    pickle.dump(results_best, handle)

with open(save_dir + model_type + "_last.pickle", 'wb') as handle:
    pickle.dump(results_last, handle)