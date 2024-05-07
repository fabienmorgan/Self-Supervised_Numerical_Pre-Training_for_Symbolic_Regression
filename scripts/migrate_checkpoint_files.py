import torch

def update_checkpoint_for_complex_model(old_checkpoint_path, new_checkpoint_path):
    # Load the old checkpoint
    old_checkpoint = torch.load(old_checkpoint_path, map_location='cpu')
    old_state_dict = old_checkpoint['state_dict']

    # Create a new state dictionary
    new_state_dict = {}

    # Prefix for the SetSkeletonEncoderLayer within SetSkeletonEncoder
    prefix = 'skeleton_enc.layers.0.'

    # Update the state dict entries
    for key, value in old_state_dict.items():
        # Only modify keys that were part of the original SetSkeletonEncoder
        if key.startswith('skeleton_enc.'):
            new_key = prefix + key[len('skeleton_enc.'):]  # Remove the old base and add the new
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value  # Preserve other components as is

    # Update the old checkpoint with the new state dictionary
    old_checkpoint['state_dict'] = new_state_dict

    # Save the updated checkpoint
    torch.save(old_checkpoint, new_checkpoint_path)
    print(f"Updated checkpoint saved to {new_checkpoint_path}")

# Usage
old_checkpoint_path = 'weights/10000000_log_-epoch=201-val_loss=0.00.ckpt'
new_checkpoint_path = 'weights/10000000_log_-epoch=201-val_loss=0.00_new_version.ckpt'
update_checkpoint_for_complex_model(old_checkpoint_path, new_checkpoint_path)
