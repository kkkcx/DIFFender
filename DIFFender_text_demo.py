import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch, logging
from PIL import Image
from torchvision import transforms as tfms
import numpy as np
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from diffusers import StableDiffusionInpaintPipeline
import time
from torch.nn import functional as F


def load_artifacts():
    time_start = time.time()
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae",
                                        torch_dtype=torch.float16).to("cuda")

    time_1 = time.time()
    print('time_1:', time_1 - time_start, 's')
    unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet",
                                                torch_dtype=torch.float16).to("cuda")

    time_2 = time.time()
    print('time_2:', time_2 - time_start, 's')
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16)
    time_3 = time.time()
    print('time_3:', time_3 - time_start, 's')
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14",
                                                 torch_dtype=torch.float16).to("cuda")
    time_4 = time.time()
    print('time_4:', time_4 - time_start, 's')

    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                              set_alpha_to_one=False)
    pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting",
                                                          revision="fp16", torch_dtype=torch.float16, ).to("cuda")
    time_5 = time.time()
    print('time_5:', time_5 - time_start, 's')
    print("done")
    return vae, unet, tokenizer, text_encoder, scheduler, pipe


def load_image(p):
    '''
    Function to load images from a defined path
    '''
    return Image.open(p).convert('RGB').resize((512, 512))


def pil_to_latents(image):
    '''
    Function to convert image to latents
    '''
    init_image = image[0].unsqueeze(0) * 2.0 - 1.0
    init_image = init_image.to(device="cuda", dtype=torch.float16)
    init_latent_dist = vae.encode(init_image).latent_dist.sample() * 0.18215
    return init_latent_dist


def latents_to_pil(latents):
    latents = (1 / 0.18215) * latents
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images


def text_enc(prompts, maxlen=None):
    if maxlen is None: maxlen = tokenizer.model_max_length
    inp = tokenizer(prompts, padding="max_length", max_length=maxlen, truncation=True, return_tensors="pt")
    return text_encoder(inp.input_ids.to("cuda"))[0].half()


vae, unet, tokenizer, text_encoder, scheduler, pipe = load_artifacts()


def prompt_2_img_i2i_fast(prompts, init_img, g=7.5, seed=100, strength=0.5, steps=50, dim=512):
    text = prompts
    uncond = text_enc([""], text.shape[1])
    emb = torch.cat([uncond, text])

    # Setting the seed
    if seed: torch.manual_seed(seed)

    # Setting number of steps in scheduler
    scheduler.set_timesteps(steps)

    # Convert the seed image to latent
    init_latents = pil_to_latents(init_img)

    # Figuring initial time step based on strength
    init_timestep = int(steps * strength)
    timesteps = scheduler.timesteps[-init_timestep]
    timesteps = torch.tensor([timesteps], device="cuda")

    # Adding noise to the latents
    noise = torch.randn(init_latents.shape, generator=None, device="cuda", dtype=init_latents.dtype)
    init_latents = scheduler.add_noise(init_latents, noise, timesteps)
    latents = init_latents

    # We need to scale the i/p latents to match the variance
    inp = scheduler.scale_model_input(torch.cat([latents] * 2), timesteps)
    # Predicting noise residual using U-Net
    # with torch.no_grad(): u, t = unet(inp, timesteps, encoder_hidden_states=emb).sample.chunk(2)
    u, t = unet(inp, timesteps, encoder_hidden_states=emb).sample.chunk(2)

    # Performing Guidance
    pred = u + g * (t - u)

    # Zero shot prediction
    latents = scheduler.step(pred, timesteps, latents).pred_original_sample

    # print('latents:', latents)

    # Returning the latent representation to output an array of 4x64x64
    return latents


# | code-fold: false
def prompt_2_img_i2i_fast_og(prompts, init_img, g=7.5, seed=100, strength=0.5, steps=50, dim=512):
    # Converting textual prompts to embedding
    text = text_enc(prompts)

    # Adding an unconditional prompt , helps in the generation process
    uncond = text_enc([""], text.shape[1])
    emb = torch.cat([uncond, text])

    # Setting the seed
    if seed: torch.manual_seed(seed)

    # Setting number of steps in scheduler
    scheduler.set_timesteps(steps)

    # Convert the seed image to latent
    init_latents = pil_to_latents(init_img)

    # Figuring initial time step based on strength
    init_timestep = int(steps * strength)
    timesteps = scheduler.timesteps[-init_timestep]
    timesteps = torch.tensor([timesteps], device="cuda")

    # Adding noise to the latents
    noise = torch.randn(init_latents.shape, generator=None, device="cuda", dtype=init_latents.dtype)
    init_latents = scheduler.add_noise(init_latents, noise, timesteps)
    latents = init_latents

    # We need to scale the i/p latents to match the variance
    inp = scheduler.scale_model_input(torch.cat([latents] * 2), timesteps)
    # Predicting noise residual using U-Net
    # with torch.no_grad(): u, t = unet(inp, timesteps, encoder_hidden_states=emb).sample.chunk(2)
    u, t = unet(inp, timesteps, encoder_hidden_states=emb).sample.chunk(2)

    # Performing Guidance
    pred = u + g * (t - u)

    # Zero shot prediction
    latents = scheduler.step(pred, timesteps, latents).pred_original_sample

    # Returning the latent representation to output an array of 4x64x64
    return latents


def create_mask_fast2(init_img, rp, ep, n=20, s=0.5):
    diff = {}

    for idx in range(n):
        print('count:', idx)
        empty_noise = prompt_2_img_i2i_fast_og(prompts=ep, init_img=init_img, strength=s, seed=100 * idx)[0]

        text_noise = prompt_2_img_i2i_fast_og(prompts=rp, init_img=init_img, strength=s, seed=100 * idx)[0]

        diff[idx] = (np.array(text_noise.detach().cpu()) - np.array(empty_noise.detach().cpu()))

        tmp = (text_noise - empty_noise).unsqueeze(0)
        if idx == 0:
            diff2 = tmp
        else:
            diff2 = torch.cat((diff2, tmp), 0)

    ## Creating a mask placeholder
    mask = np.zeros_like(diff[0])
    mask_t = torch.zeros(diff2[0].shape).to("cuda")

    ## Taking an average of m iterations
    for idx in range(n):
        ## Note np.abs is a key step
        mask += np.abs(diff[idx])

        mask_t += torch.abs(diff2[idx])

        ## Averaging multiple channels
    mask_t = torch.mean(mask_t, 0)

    ## Normalizing

    mask = (mask_t - torch.mean(mask_t)) / torch.std(mask_t)

    mask_max = mask_t.max()
    mask_min = mask_t.min()
    mask_t = (mask_t - mask_min) / (mask_max - mask_min)

    mask = mask + ((mask > 2).int() - mask).detach()

    return mask, mask_t


def improve_mask2(mask):
    mask = mask.unsqueeze(0)
    GaussianBlur = tfms.GaussianBlur((3, 3), sigma=1)
    mask = GaussianBlur(mask)
    return mask


def DIFFender(init_img, rp, qp, g=7.5, seed=100, strength=0.5, steps=20, dim=512):

    ep = [""]
    l_batch = 1

    for i in range(l_batch):
        mask, mask_n = create_mask_fast2(init_img=[init_img[i]], rp=rp, ep=ep, n=3)
        mask = improve_mask2(mask)

        # max_pool2d
        kernel_size = 3
        padding = (kernel_size - 1) // 2
        mask = F.max_pool2d(mask, kernel_size, stride=1, padding=padding)

        tmp = mask_n.unsqueeze(0)
        if i == 0:
            M_n = tmp
        else:
            M_n = torch.cat((M_n, tmp), 0)

        tmp2 = mask.unsqueeze(0)
        if i == 0:
            M_n2 = tmp2
        else:
            M_n2 = torch.cat((M_n2, tmp2), 0)

    mask = M_n2
    mask_n = M_n

    qp = qp * l_batch

    from torchvision.transforms import Resize
    torch_resize = Resize([512, 512])  # 定义Resize类对象
    mask = torch_resize(mask)

    print("init_img.shape:",init_img.shape)
    print("mask:", mask.shape)


    output = pipe(
        prompt=qp,
        # prompt_embeds=qp,
        image=init_img,
        mask_image=mask.to("cuda"),
        generator=torch.Generator("cuda").manual_seed(100),
        num_inference_steps=20
    ).images

    return mask, output


text_1 = "enter_your_text1"
text_2 = "enter_your_text2"

path = 'enter_your_path'
init_img = load_image(path)

init_img = [tfms.Compose([tfms.ToTensor()])(init_img)]
init_img = torch.stack(init_img).to("cuda")

mask, output = DIFFender(init_img, rp=text_1, qp=text_2)

from torchvision.utils import save_image

save_image(output, 'output.png')
save_image(mask, 'mask.png')




