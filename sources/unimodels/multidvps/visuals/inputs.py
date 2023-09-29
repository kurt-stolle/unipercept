from einops import rearrange


def visualize_images(images):
    images = rearrange(images, "b c h w -> c h (b w)")
    yield "inputs", images
