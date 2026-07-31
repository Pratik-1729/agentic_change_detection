import torch


def load_changeformer(
    model_builder,
    checkpoint_path,
    device
):

    model = model_builder()

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint
    )

    model.to(device)

    model.eval()

    return model