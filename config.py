import sys

sys.path.append("../nel_framework/nel/")
import nel
import random

# ---------------- config 1
items1 = []
items1.append(nel.Item("banana", [1.0, 1.0, 0.0], [1.0, 1.0, 0.0], True))

intensity_fn_args1 = [-2.0]
interaction_fn_args1 = [len(items1)]
interaction_fn_args1.extend([40.0, 200.0, 0.0,
                             -40.0])  # parameters for interaction between item 0 and item 0

config1 = nel.SimulatorConfig(
    max_steps_per_movement=1, vision_range=4,
    patch_size=32, gibbs_num_iter=10, items=items1,
    agent_color=[0.0, 0.0, 1.0],
    collision_policy=nel.MovementConflictPolicy.FIRST_COME_FIRST_SERVED,
    decay_param=0.4, diffusion_param=0.14,
    deleted_item_lifetime=2000,
    intensity_fn=nel.IntensityFunction.CONSTANT,
    intensity_fn_args=intensity_fn_args1,
    interaction_fn=nel.InteractionFunction.PIECEWISE_BOX,
    interaction_fn_args=interaction_fn_args1)



#config2 = nel.SimulatorConfig(seed=random.randint(0,4294967295),
#    max_steps_per_movement=1, vision_range=5,
#    patch_size=32, gibbs_num_iter=10, items=items,
#    agent_color=[1.0, 0.5, 0.5],
#    collision_policy=nel.MovementConflictPolicy.FIRST_COME_FIRST_SERVED,
#    decay_param=0.4, diffusion_param=0.14,
#    deleted_item_lifetime=2000,
#    intensity_fn=nel.IntensityFunction.CONSTANT,
#    intensity_fn_args=intensity_fn_args,
#    interaction_fn=nel.InteractionFunction.PIECEWISE_BOX,
#    interaction_fn_args=interaction_fn_args)

agent_config = {
    'history_len': 2,
    'learning_rate': 1e-4}

train_config={
    'discount_factor': .99,
    'eval_steps': 1000,
    'max_steps': 500001,
    'batch_size': 64,
    'eval_frequency': 1000,
    'target_update_frequency': 1000,
    'policy_update_frequency': 2,
    'replay_buffer_capacity': 10000,
    'num_steps_save_training_run': 100000
    }
