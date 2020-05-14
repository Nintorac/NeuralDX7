#%%
from os import environ
environ['MLFLOW_TRACKING_URI'] = 'http://tracking.olympus.nintorac.dev:9001/'

from neuralDX7.constants import N_PARAMS, MAX_VALUE
from agoge.utils import trial_name_creator
from neuralDX7 import DEFAULTS
from agoge import TrainWorker as Worker
from ray import tune
from neuralDX7.models import DX7VAE as Model
from neuralDX7.solvers import DX7VAE as Solver
from neuralDX7.datasets import DX7SysexDataset as Dataset

def config(experiment_name, trial_name, 
        n_heads=8, n_features=64, 
        batch_size=16, data_size=1.,
        latent_dim=8, num_flows=16,
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
        'hidden_dim': layer_features * 3
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
        'latent_dim': latent_dim,
        'encoder': encoder,
        'decoder': {
            'c_features': latent_dim,
            'features': layer_features,
            'attention_layer': attention_layer,
            'max_len': N_PARAMS,
            'n_layers': 12
        },
        'num_flows': num_flows,
        'deterministic_path_drop_rate': 0.8
    }

    solver = {
        'Solver': Solver,
        'beta_temp': 6e-5,
        'max_beta': 0.5
    }

    tracker = {
        'metrics': [
            'reconstruction_loss', 
            'accuracy', 
            'kl', 
            'beta', 
            'log_det', 
            'q_z_0',
            'p_z_k',
        ],
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
    postfix = sys.argv[1] if len(sys.argv)==2 else ''
    # ray.init()
    # from ray.tune.utils import validate_save_restore
    # validate_save_restore(Worker)
    # client = MlflowClient(tracking_uri='localhost:5000')
    experiment_name = f'dx7-vae-dev'#+experiment_name_creator()
    # experiment_id = client.create_experiment(experiment_name)


    experiment_metrics = dict(metric="loss/accuracy", mode="max")

    tune.run(Worker, 
    config={
        'config_generator': config,
        'experiment_name': experiment_name,
        'points_per_epoch': 10
    },
    trial_name_creator=trial_name_creator,
    resources_per_trial={
        # 'gpu': 1
        # 'cpu': 5
    },
    checkpoint_freq=2,
    checkpoint_at_end=True,
    keep_checkpoints_num=1,
    # search_alg=bohb_search, 
    # scheduler=bohb_hyperband,
    num_samples=1,
    verbose=0,
    local_dir=DEFAULTS['ARTIFACTS_ROOT']
    # webui_host='127.0.0.1' ## supresses an error
        # stop={'loss/loss': 0}
    )
# points_per_epoch

# %%
