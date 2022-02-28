import numpy as np
import jax.numpy as jnp
import os
import wandb
import warnings


def slurm_infos():
    return {
        'slurm/job_id': os.getenv('SLURM_JOB_ID'),
        'slurm/job_user': os.getenv('SLURM_JOB_USER'),
        'slurm/job_partition': os.getenv('SLURM_JOB_PARTITION'),
        'slurm/cpus_per_node': os.getenv('SLURM_JOB_CPUS_PER_NODE'),
        'slurm/num_nodes': os.getenv('SLURM_JOB_NUM_NODES'),
        'slurm/nodelist': os.getenv('SLURM_JOB_NODELIST'),
        'slurm/cluster_name': os.getenv('SLURM_CLUSTER_NAME'),
        'slurm/array_task_id': os.getenv('SLURM_ARRAY_TASK_ID')
    }


def table_from_array(array):
    data = [(value,) + index for (index, value) in np.ndenumerate(array)]
    columns = ['data'] + [f'x_{i}' for i in range(array.ndim)]
    return wandb.Table(data=data, columns=columns)


def to_wandb(logs, prefix=None, keys=None):
    converted_logs = {}
    prefix = '' if (prefix is None) else f'{prefix}/'
    for key, value in logs.items():
        if (keys is not None) and (key not in keys):
            continue

        if np.ndim(value) == 0:
            converted_logs[f'{prefix}{key}'] = value

        elif isinstance(value, (np.ndarray, jnp.DeviceArray)):
            converted_logs[f'{prefix}{key}'] = table_from_array(value)

        else:
            warnings.warn(f'Unknown type {type(value)} for key: {key}')
    return converted_logs
