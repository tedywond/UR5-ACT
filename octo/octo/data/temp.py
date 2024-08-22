def get_episode_dict(
        image, # (steps, 256, 256, 3)
        action, # (steps, DOF)
        endeffect_state, # (steps, DOF)
        language_instr, # str
        dataset_name, # str
        absolute_action_mask = [False] * 6 + [True] # (DOF, )
        ):
    assert image.shape[0] == action.shape[0] == endeffect_state.shape[0]
    # TODO: what do we do if the episodes are not the same length?

    n_steps = image.shape[0]
    n_dof = action.shape[1]

    image = np.expand_dims(image, axis=1)

    action_tensor = np.zeros((n_steps, 50, n_dof))
    padding = np.zeros((50, n_dof))
    for i in range(n_dof):
        if absolute_action_mask[i]:
            padding[:, i] = action[-1, i]

    action = np.concatenate([action, padding], axis=0)
    for i in range(n_steps):
        action_tensor[i, :] = action[i:i+50, :]

    language_instr = np.array(language_instr.encode('utf-8'))
    language_instr = np.repeat(language_instr, n_steps)

    dataset_name = np.array(dataset_name)
    dataset_name = np.repeat(dataset_name, n_steps)

    endeffect_state = np.expand_dims(endeffect_state, axis=1)

    timestep = np.arange(n_steps)

    absolute_action_mask = np.array(absolute_action_mask)
    absolute_action_mask = np.repeat(absolute_action_mask[np.newaxis, :], n_steps, axis=0)

    episode_dict = {
        'image': image,
        'endeffector_state': endeffect_state,
        'timestep': timestep,
        'action': action_tensor,
        'language_instr': language_instr,
        'dataset_name': dataset_name,
        'absolute_action_mask': absolute_action_mask,
        'pad_mask': np.ones((n_steps, 1), dtype=bool),
        'pad_mask_image': np.ones((n_steps, 1), dtype=bool),
        'pad_mask_endeffector_state': np.ones((n_steps, 1), dtype=bool),
        'pad_mask_timestep': np.ones((n_steps, 1), dtype=bool),
        'pad_mask_language_instr': np.ones((n_steps,), dtype=bool),
    }

    return episode_dict
