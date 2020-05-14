#%%
from os import environ
environ['MLFLOW_TRACKING_URI'] = 'http://tracking.olympus.nintorac.dev:9001/'
# environ['MLFLOW_TRACKING_URI'] = 'http://localhost:9001/'
#environ['ARTIFACTS_ROOT'] = '/content/gdrive/My Drive/audio/artifacts'
# ARTIFACTS_ROOT='/content/gdrive/My Drive/audio/artifacts'
from neuralDX7.constants import N_PARAMS, MAX_VALUE
from agoge.utils import trial_name_creator
from neuralDX7 import DEFAULTS
from agoge import TrainWorker as Worker
from ray import tune
from neuralDX7.models import DX7PatchProcess as Model
from neuralDX7.solvers import DX7PatchProcess as Solver
from neuralDX7.datasets import DX7SysexDataset as Dataset

def config(experiment_name, trial_name, 
        n_heads=8, n_features=32, 
        batch_size=16, data_size=0.05,
        **kwargs):
    


    data_handler = {
        'Dataset': Dataset,
        'dataset_opts': {
            'data_size': data_size
        },
        'loader_opts': {
            'batch_size': batch_size,
        },
    }

    ### MODEL FEATURES
    layer_features = n_heads * n_features

    head_features = layer_features // n_heads

    attention = {
        'n_features': layer_features,
        'n_hidden': head_features,
        'n_heads': n_heads
    }
    
    attention_layer = {
        'attention': attention,
        'features': layer_features,
        'hidden_dim': layer_features * 2
    }

    encoder = {
        'features': layer_features,
        'attention_layer': attention_layer,
        'max_len': N_PARAMS,
        'n_layers': 12
    }
    

    model = {
        'Model': Model,
        'features': layer_features,
        'encoder': encoder
    }

    solver = {
        'Solver': Solver,
        'lr': 1e-3,
    }

    tracker = {
        'metrics': ['reconstruction_loss', 'accuracy'],
        'experiment_name': experiment_name,
        'trial_name': trial_name
    }

    return {
        'data_handler': data_handler,
        'model': model,
        'solver': solver,
        'tracker': tracker,
    }

if __name__=='__main__':
    # from ray import ray
    import sys
    import mlflow
    from mlflow.tracking import MlflowClient
    postfix = sys.argv[1] if len(sys.argv)==2 else ''

    # ray.init()
    # from ray.tune.utils import validate_save_restore
    # validate_save_restore(Worker)
    client = MlflowClient()
    experiment_name = f'dx7-vae-{postfix}'#+experiment_name_creator()
    resume=False
    try:
        experiment_id = client.create_experiment(experiment_name)
    except mlflow.exceptions.RestException:
        resume = True

    experiment_metrics = dict(metric="loss/accuracy", mode="max")
    import torch
    gpus = 0.5 if torch.cuda.is_available() else 0
    gpus = 1
    
    # import ray

    # ray.init()
    # ray.tune.utils.validate_save_restore(Worker)

    tune.run(Worker, 
    config={
        'config_generator': config,
        'experiment_name': experiment_name,
        'points_per_epoch': 10
    },
    trial_name_creator=trial_name_creator,
    resources_per_trial={
        # 'gpu': gpus,
        'cpu': 6
    },
    checkpoint_freq=2,
    checkpoint_at_end=True,
    keep_checkpoints_num=1,
    # search_alg=bohb_search, 
    # scheduler=bohb_hyperband,
    num_samples=1,
    verbose=1,
    local_dir=DEFAULTS['ARTIFACTS_ROOT'],
    resume=resume
    # webui_host='127.0.0.1' ## supresses an error
        # stop={'loss/loss': 0}
    )
# points_per_epoch
