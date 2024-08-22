import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os
import json
from PIL import Image
from scipy.spatial.transform import Rotation

"""
<DLataset element_spec={
    'observation': {
        'image_primary': TensorSpec(shape=(steps, 1, 256, 256, 3), dtype=tf.uint8, name=None),
        'proprio': TensorSpec(shape=(steps, 1, dof), dtype=tf.float32, name=None),
        'timestep': TensorSpec(shape=(steps, 1), dtype=tf.int32, name=None),
        'pad_mask_dict': {
            'image_primary': TensorSpec(shape=(steps, 1), dtype=tf.bool, name=None),
            'proprio': TensorSpec(shape=(steps, 1), dtype=tf.bool, name=None),
            'timestep': TensorSpec(shape=(steps, 1), dtype=tf.bool, name=None)},
                NOTE: This is False if the dataset does not have these observations
                These are all true for the ALOHA example episode
        'pad_mask': TensorSpec(shape=(steps, 1), dtype=tf.bool, name=None)},
            NOTE: When the observation window size is 1, the pad_mask is all True
            This is all true for the ALOHA example episode
    'task': {
        'language_instruction': TensorSpec(shape=(steps,), dtype=tf.string, name=None),
        'pad_mask_dict': {
            'language_instruction': TensorSpec(shape=(steps,), dtype=tf.bool, name=None)}},
                NOTE: This should be False if the dataset has no language instructions
    'action': TensorSpec(shape=(steps, 50, dof), dtype=tf.float32, name=None),
        NOTE: For relative actions, the timesteps after the last action should be zero-padded
        For absolute actions, the timesteps after the last action should be the last action repeated
    'dataset_name': TensorSpec(shape=(steps,), dtype=tf.string, name=None),
    'absolute_action_mask': TensorSpec(shape=(steps, dof), dtype=tf.bool, name=None)}>
        NOTE: This tell whether a dimension of the action is relative (dx,dy,dz) or absolute (x,y,z)
"""

"""
When recording joint states from the UR5 arm, some of the joint states are zeros.
This function interpolates the null entries with the nearest non-null entry.
"""
def interpolate_null_entries(data_dict):
    episode_length = len(data_dict['states'])
    if len(data_dict['joints'][0]['joint_states']) is not 6:
        for i in range(1, episode_length):
            if len(data_dict['joints'][i]['joint_states']) is 6:
                data_dict['joints'][0]['joint_states'] = data_dict['joints'][i]['joint_states']
                break
    for i in range(episode_length):
        if len(data_dict['joints'][i]['joint_states']) is not 6:
            for j in range(i - 1, 0, -1):
                if len(data_dict['joints'][j]['joint_states']) is 6:
                    data_dict['joints'][i]['joint_states'] = data_dict['joints'][j]['joint_states']
                    break

    return data_dict


def create_dataset(create_from='0215', dataset_name='0222_joints'):
    files = os.listdir(create_from)
    dataset_name = '0222_joints'
    language_instruction = 'Put the yellow block in the wooden box'.encode('utf-8')
    planning_horizon = 4

    data_dicts = []
    for file in files:
        data_dict = json.load(open(f'0215/{file}'))
        data_dicts.append(data_dict)

    # max_length = max([len(data_dict['states']) for data_dict in data_dicts])

    step_dicts = []
    for data_dict in data_dicts:
        print(f'Processing {data_dict["episode"]}...')
        episode_length = len(data_dict['states'])
        proprios = []
        actions = []
        images = []

        data_dict = interpolate_null_entries(data_dict)

        for i in range(episode_length):
            # euler = list(Rotation.from_quat(data_dict['states'][i]['rotation']).as_euler('xyz', degrees=False))
            # proprios.append(np.array(
            #     data_dict['states'][i]['translation']
            #     + euler
            #     + [data_dict['states'][i]['gripper_state']]))

            proprios.append(np.array(
                data_dict['joints'][i]['joint_states'] +
                [data_dict['states'][i]['gripper_state']]
            ))

            image_path = f'images/{file.split(".")[0]}_side/{i}.png'
            try:
                image = Image.open(image_path)
                image = image.resize((256, 256))
                image = np.array(image)
            except:
                image = np.zeros((256, 256, 3), dtype=np.uint8)
            images.append(image)

        # The actions are obtain by taking the difference between states in adjacent timesteps
        for i in range(episode_length - 1):
            actions.append(np.concatenate([proprios[i+1][:6] - proprios[i][:6], [data_dict['states'][i+1]['gripper_state']]]))
            # actions.append(np.concatenate([proprios[i+1][:6], [data_dict['states'][i+1]['gripper_state']]]))

        for i in range(planning_horizon + 1):
            actions.append(actions[-1])

        try:
            actions = np.array(actions)
        except:
            import pdb; pdb.set_trace()

        for i in range(episode_length):
            step_dict = {
                'image_primary': np.expand_dims(images[i], (0, 1)),
                'proprio': np.expand_dims(proprios[i], (0, 1)),
                'timestep': np.array([[i]]),
                'action': np.expand_dims(actions[i+1:i+planning_horizon+1], 0),
                'language_instruction': np.array([language_instruction]),
                'dataset_name': np.array([dataset_name]),
                'absolute_action_mask': np.array([[False] * 6 + [True]]),
                'pad_mask': np.array([[True]]),
                'pad_mask_image': np.array([[True]]),
                'pad_mask_endeffector_state': np.array([[True]]),
                'pad_mask_timestep': np.array([[True]]),
                'pad_mask_language_instr': np.array([True]),
            }
            step_dicts.append(step_dict)

    dataset = {key: [step_dict[key] for step_dict in step_dicts] for key in step_dicts[0]}

    # normalize actions and states
    statistics = {}
    actions = np.array(dataset['action'])
    statistics["action_mean"] = np.mean(actions, axis=(0, 1, 2))
    statistics["action_std"] = np.std(actions, axis=(0, 1, 2))
    actions = (actions - statistics["action_mean"]) / statistics["action_std"]
    dataset['action'] = actions

    endeffector_states = np.array(dataset['proprio'])
    statistics["state_mean"] = np.mean(endeffector_states, axis=(0, 1))
    statistics["state_std"] = np.std(endeffector_states, axis=(0, 1))
    endeffector_states = (endeffector_states - statistics["state_mean"]) / statistics["state_std"]
    dataset['proprio'] = endeffector_states

    dataset = tf.data.Dataset.from_tensor_slices(dataset)

    # check if /data exists
    if not os.path.exists('/datasets/0222_joints'):
        os.makedirs('/datasets/0222_joints')

    tf.data.Dataset.save(dataset, 'datasets/0222_joints')

    for key in statistics:
        statistics[key] = statistics[key].tolist()
    json.dump(statistics, open('datasets/0222_joints/statistics.json', 'w'))

if __name__ == '__main__':
    create_dataset()
    # reloaded = tf.data.Dataset.load('datasets/0220')

    # python packages/octo/octo/data/create_dataset.py