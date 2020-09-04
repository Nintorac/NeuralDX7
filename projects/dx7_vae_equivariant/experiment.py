#%%
from os import environ
environ['MLFLOW_TRACKING_URI'] = 'http://tracking.olympus.nintorac.dev:9001/'

from neuralDX7.constants import N_PARAMS, MAX_VALUE
from agoge.utils import trial_name_creator
from neuralDX7 import DEFAULTS
from agoge import TrainWorker as Worker
from ray import tune
from neuralDX7.models import DX7VAEEquivariant as Model
from neuralDX7.solvers import DX7VAE as Solver
from neuralDX7.datasets import SingleVoiceLMDBDataset as Dataset

def config(experiment_name, trial_name, 
        param_n_layers=3,
        param_n_heads=8, param_n_features=64, 
        param_batch_size=16, data_size=1.,
        param_latent_dim=8, param_num_flows=16,
        param_equivariant_layer_type='linear',
        **kwargs):
    


    data_handler = {
        'Dataset': Dataset,
        'dataset_opts': {
            'keys_file': 'unique_voice_keys.npy',
            'data_file': 'dx7-data.lmdb',
            'root':'~/data',
            'data_size': 1.
        },
        'loader_opts': {
            'batch_size': param_batch_size,
            "shuffle": True
        },
    }
    }

    ### MODEL FEATURES
    equivariant_layer = {
        'features': param_n_features,
        'feed_forward': {
            'features': param_n_features
        },
        'core': 'linear',
        'core_args':{
            'input_features': param_n_features,
            'output_features': param_n_features,
            'n_heads': param_n_heads
        }
    }

    invariant_layer = {
        'features': param_n_features,
        'core': 'invariant',
        'feed_forward': {
            'features': param_n_features
        },
        'core_args':{
            'input_features': param_n_features,
            'output_features': param_n_features,
            'n_heads': param_n_heads,
            'max_len': 155
        }
    }
    
    equivariant_list = [equivariant_layer] * (param_n_layers-1)

    encoder_stack = {
        'layer_args': [invariant_layer] + equivariant_list,
        'features': param_n_features,

    }

    decoder_stack = {
        'layer_args': equivariant_list + [invariant_layer],
        'features': param_n_features,
        'conditioning_dim': param_latent_dim

    }
    

    model = {
        'Model': Model,
        'features': param_n_features,
        'latent_dim': param_latent_dim,
        'encoder': encoder_stack,
        'decoder': decoder_stack,
        'num_flows': param_num_flows,
    }

    solver = {
        'Solver': Solver,
        'beta_temp': 6e-5,
        'max_beta': 0.5
    }


    return {
        'data_handler': data_handler,
        'model': model,
        'solver': solver,
    }

if __name__=='__main__':
    # from ray import ray
    import sys
    dev=True
    # ray.init()
    # from ray.tune.utils import validate_save_restore
    # validate_save_restore(Worker)
    # client = MlflowClient(tracking_uri='localhost:5000')
    experiment_name = f'dx7-vae-equivariant'#+experiment_name_creator()
    # experiment_id = client.create_experiment(experiment_name)

    ray_config = {
        'config_generator': config,
        'experiment_name': experiment_name,
        'points_per_epoch': 10,

        'param_n_layers': 8,
        'param_n_heads': 8,
        'param_n_features': 128,
        'param_batch_size': 64,
        'param_latent_dim': 32,
        'param_num_flows': 0,
        'param_equivariant_layer_type': 'linear',
    }
    if dev:
        ray_config.update({'trial_name': 'dev'})
        worker = Worker(config=ray_config)
        worker.epoch(worker.dataset.loaders.test, 'test')
        1/0

    experiment_metrics = dict(metric="loss/accuracy", mode="max")

    tune.run(Worker, name=experiment_name,
    config=ray_config,
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




# %%
