
def hard_target_update(local, target):
        target.load_state_dict(local.state_dict())

def soft_target_update(local, target, tau):
    for target_param, local_param in zip(target.parameters(), local.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
