import asyncio
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from diffusers_helper.hf_login import login

import os

os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))

import gradio as gr
import torch
import traceback
import einops
import safetensors.torch as sf
import numpy as np
import argparse
import math
import time # Make sure time is imported
import atexit # For pynvml cleanup
import random # For random seed

# --- GPU Monitor Imports ---
try:
    import pynvml
    pynvml_available = True
except ImportError:
    pynvml_available = False
    print("Warning: pynvml not found. GPU stats will not be displayed. Install with: pip install nvidia-ml-py")
# --- End GPU Monitor Imports ---


from PIL import Image
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, state_dict_weighted_merge, state_dict_offset_merge, generate_timestamp
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
# Import necessary functions including the new unload_all_models
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete, unload_all_models
from diffusers_helper.thread_utils import AsyncStream, async_run
from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html
from transformers import SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket


# --- GPU Monitor Helper Function ---
def format_bytes(size_bytes):
    """Converts bytes to a human-readable format (MiB or GiB)."""
    if size_bytes is None or size_bytes == 0:
        return "0 MiB"
    size_mib = size_bytes / (1024**2)
    if size_mib >= 1024:
        size_gib = size_mib / 1024
        return f"{size_gib:.1f} GiB"
    else:
        # Display MiB with no decimal places for compactness
        return f"{size_mib:.0f} MiB"

# --- GPU Monitor Initialization ---
nvml_initialized = False
gpu_handle = None
gpu_name = "N/A" # Store GPU name here
if pynvml_available:
    try:
        print("Initializing NVML for GPU monitoring...")
        pynvml.nvmlInit()
        nvml_initialized = True
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count > 0:
            gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0) # Get handle for GPU 0
            gpu_name = pynvml.nvmlDeviceGetName(gpu_handle) # Get GPU name
            # Try to decode if it's bytes (common in pynvml)
            if isinstance(gpu_name, bytes):
                gpu_name = gpu_name.decode('utf-8')
            # Optional: Shorten common names like "NVIDIA GeForce RTX 3090" -> "RTX 3090"
            gpu_name = gpu_name.replace("NVIDIA GeForce ", "")
            print(f"NVML Initialized. Monitoring GPU 0: {gpu_name}")
        else:
            print("NVML Initialized, but no NVIDIA GPUs detected.")
            gpu_handle = None # Explicitly set to None
            gpu_name = "No NVIDIA GPU"
            nvml_initialized = False # Treat as not initialized if no GPUs
        # Register shutdown hook
        atexit.register(pynvml.nvmlShutdown)
        print("Registered NVML shutdown hook.")
    except pynvml.NVMLError as error:
        print(f"Failed to initialize NVML: {error}. GPU stats disabled.")
        nvml_initialized = False
        gpu_handle = None
        gpu_name = "NVML Init Error"
# --- End GPU Monitor Initialization ---


parser = argparse.ArgumentParser()
parser.add_argument('--share', action='store_true')
parser.add_argument("--server", type=str, default='0.0.0.0')
parser.add_argument("--port", type=int, required=False)
parser.add_argument("--inbrowser", action='store_true')
args = parser.parse_args()

# for win desktop probably use --server 127.0.0.1 --inbrowser
# For linux server probably use --server 127.0.0.1 or do not use any cmd flags
print(args)

free_mem_gb = get_cuda_free_memory_gb(gpu)
high_vram = free_mem_gb > 60 # Threshold for high VRAM mode

print(f'Free VRAM {free_mem_gb:.2f} GB')
print(f'High-VRAM Mode: {high_vram}')

# --- Constants ---
DEFAULT_SEED = 31337
MAX_SEED = 2**32 - 1 # Max value for typical 32-bit unsigned integer seed
SEED_MODE_LAST = 'last'
SEED_MODE_RANDOM = 'random'

# Load models initially to CPU
print("Loading models to CPU...")
text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=torch.float16).cpu()
text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()

feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()

transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePackI2V_HY', torch_dtype=torch.bfloat16).cpu()
print("Models loaded.")

# Keep a list of all core models for easier management
all_core_models = [text_encoder, text_encoder_2, vae, image_encoder, transformer]

# Set models to evaluation mode
vae.eval()
text_encoder.eval()
text_encoder_2.eval()
image_encoder.eval()
transformer.eval()

# Configure models based on VRAM
if not high_vram:
    print("Low VRAM: Enabling VAE slicing/tiling.")
    vae.enable_slicing()
    vae.enable_tiling()

transformer.high_quality_fp32_output_for_inference = True
print('transformer.high_quality_fp32_output_for_inference = True')

# Set dtypes (already done during loading, but good practice)
transformer.to(dtype=torch.bfloat16)
vae.to(dtype=torch.float16)
image_encoder.to(dtype=torch.float16)
text_encoder.to(dtype=torch.float16)
text_encoder_2.to(dtype=torch.float16)

# Disable gradients
vae.requires_grad_(False)
text_encoder.requires_grad_(False)
text_encoder_2.requires_grad_(False)
image_encoder.requires_grad_(False)
transformer.requires_grad_(False)

# Apply memory optimization strategies based on VRAM
if not high_vram:
    # DynamicSwapInstaller for low VRAM
    print("Low VRAM mode: Enabling DynamicSwap for Transformer and Text Encoder.")
    DynamicSwapInstaller.install_model(transformer, device=gpu)
    DynamicSwapInstaller.install_model(text_encoder, device=gpu)
else:
    # Preload models to GPU for high VRAM
    print("High VRAM mode: Preloading models to GPU.")
    try:
        text_encoder.to(gpu)
        text_encoder_2.to(gpu)
        image_encoder.to(gpu)
        vae.to(gpu)
        transformer.to(gpu)
        print("Models successfully moved to GPU.")
    except Exception as e:
         print(f"Error moving models to GPU: {e}. Check available VRAM.")
         # Consider falling back to low VRAM settings or exiting
         high_vram = False # Fallback to low VRAM behavior if loading fails
         print("Falling back to Low VRAM mode settings.")
         DynamicSwapInstaller.install_model(transformer, device=gpu)
         DynamicSwapInstaller.install_model(text_encoder, device=gpu)


stream = AsyncStream()

outputs_folder = './outputs/'
os.makedirs(outputs_folder, exist_ok=True)


@torch.no_grad()
def worker(input_image, end_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf , resolution, keep_models_alive, keep_only_final_video):
    """The main generation worker function."""
    job_id = generate_timestamp()
    output_filename = None # Initialize output filename
    previous_output_filename = None # To track intermediate files for deletion
    dynamic_swap_installed = False # Track if DynamicSwap was installed for cleanup

    try:
        # Seed is already determined by the 'process' function based on the mode
        current_seed = int(seed)
        print(f"Worker using Seed: {current_seed}")

        # Determine number of sections
        total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
        total_latent_sections = int(max(round(total_latent_sections), 1))
        print(f"Planned generation: {total_second_length} seconds, {total_latent_sections} sections.")

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting generation...'))))

        # Initial model setup based on VRAM mode
        if not high_vram:
             # Ensure DynamicSwap is installed if not already (e.g., first run)
             if not hasattr(transformer, 'forge_backup_original_class'):
                 print("Applying DynamicSwap (worker check)...")
                 DynamicSwapInstaller.install_model(transformer, device=gpu)
                 DynamicSwapInstaller.install_model(text_encoder, device=gpu)
                 dynamic_swap_installed = True # Mark for cleanup
             else:
                 dynamic_swap_installed = True # Assume it's already installed if attribute exists

        # --- Text Encoding ---
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Encoding text prompt...'))))
        if not high_vram:
            # fake_diffusers handles text_encoder due to DynamicSwap
            fake_diffusers_current_device(text_encoder, gpu)
            # text_encoder_2 needs explicit loading
            load_model_as_complete(text_encoder_2, target_device=gpu, unload=True) # unload=True unloads previous 'complete' models
        else:
            # Ensure models are on GPU in high VRAM mode
            text_encoder.to(gpu)
            text_encoder_2.to(gpu)

        llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
        if cfg == 1:
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)
        print("Text encoding complete.")

        # --- Image Processing ---
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Processing start frame...'))))
        H, W, C = input_image.shape
        height, width = find_nearest_bucket(H, W, resolution=resolution)
        print(f"Input resolution {W}x{H}, selected bucket {width}x{height}")
        input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)
        Image.fromarray(input_image_np).save(os.path.join(outputs_folder, f'{job_id}_start.png'))
        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1.0
        input_image_pt = input_image_pt.permute(2, 0, 1).unsqueeze(0).unsqueeze(2) # Shape: [1, C, 1, H, W]
        print(f"Start frame processed. Shape: {input_image_pt.shape}")

        has_end_image = end_image is not None
        if has_end_image:
            stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Processing end frame...'))))
            H_end, W_end, C_end = end_image.shape
            # Ensure end image uses the same bucket dimensions
            end_image_np = resize_and_center_crop(end_image, target_width=width, target_height=height)
            Image.fromarray(end_image_np).save(os.path.join(outputs_folder, f'{job_id}_end.png'))
            end_image_pt = torch.from_numpy(end_image_np).float() / 127.5 - 1.0
            end_image_pt = end_image_pt.permute(2, 0, 1).unsqueeze(0).unsqueeze(2) # Shape: [1, C, 1, H, W]
            print(f"End frame processed. Shape: {end_image_pt.shape}")

        # --- VAE Encoding ---
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'VAE encoding frames...'))))
        if not high_vram:
            load_model_as_complete(vae, target_device=gpu, unload=True)
        else:
            vae.to(gpu)

        start_latent = vae_encode(input_image_pt, vae)
        print(f"Start latent encoded. Shape: {start_latent.shape}")
        if has_end_image:
            end_latent = vae_encode(end_image_pt, vae)
            print(f"End latent encoded. Shape: {end_latent.shape}")
        else:
            end_latent = None # Explicitly None if no end image

        # --- CLIP Vision Encoding ---
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'CLIP Vision encoding...'))))
        if not high_vram:
            load_model_as_complete(image_encoder, target_device=gpu, unload=True)
        else:
            image_encoder.to(gpu)

        image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

        if has_end_image:
            end_image_encoder_output = hf_clip_vision_encode(end_image_np, feature_extractor, image_encoder)
            end_image_encoder_last_hidden_state = end_image_encoder_output.last_hidden_state
            # Simple average for combining embeddings
            image_encoder_last_hidden_state = (image_encoder_last_hidden_state + end_image_encoder_last_hidden_state) / 2.0
            print("Combined start/end frame CLIP Vision embeddings.")
        else:
            print("Using start frame CLIP Vision embedding.")

        # --- Prepare Embeddings for Transformer ---
        llama_vec = llama_vec.to(transformer.dtype)
        llama_vec_n = llama_vec_n.to(transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)

        # --- Sampling Loop ---
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Preparing for sampling...'))))
        # Use the integer seed determined by 'process'
        rnd = torch.Generator("cpu").manual_seed(current_seed)
        num_frames_in_window = latent_window_size * 4 - 3 # Frames generated per window pass

        # History tensors kept on CPU to save VRAM, moved to GPU as needed
        history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32, device=cpu)
        history_pixels = None
        total_generated_latent_frames = 0

        # Define padding sequence
        latent_paddings = list(reversed(range(total_latent_sections)))
        if total_latent_sections > 4:
            # Apply the padding trick for longer videos
            latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]
        print(f"Using latent padding sequence: {latent_paddings}")

        for i, latent_padding in enumerate(latent_paddings):
            section_index = i + 1
            is_last_section = (latent_padding == 0)
            is_first_section = (i == 0) # Check based on loop index
            latent_padding_size = latent_padding * latent_window_size

            if stream.input_queue.top() == 'end':
                print("Stop signal received. Ending generation early.")
                stream.output_queue.push(('end', output_filename)) # Push potentially intermediate file
                return

            print(f"\n--- Starting Section {section_index}/{len(latent_paddings)} ---")
            print(f"Padding: {latent_padding} ({latent_padding_size} frames), Last: {is_last_section}, First: {is_first_section}")

            # Calculate indices for the current window
            total_indices = sum([1, latent_padding_size, latent_window_size, 1, 2, 16])
            indices = torch.arange(0, total_indices, device=cpu).unsqueeze(0)
            clean_latent_indices_pre, _, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = \
                indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

            # Prepare clean latents for conditioning (start/end frames and history)
            # Ensure start_latent is on the correct device and dtype for concatenation
            clean_latents_pre = start_latent.to(device=cpu, dtype=history_latents.dtype)

            # Get conditioning latents from history (which is on CPU)
            # Split history *before* potentially moving parts to GPU
            current_history_post, current_history_2x, current_history_4x = \
                history_latents[:, :, :1 + 2 + 16].split([1, 2, 16], dim=2)

            # Use end latent for the 'post' conditioning if available and it's the first section
            if has_end_image and is_first_section:
                 print("Using end latent for conditioning in the first section.")
                 clean_latents_post = end_latent.to(device=cpu, dtype=history_latents.dtype)
            else:
                 clean_latents_post = current_history_post

            # Combine pre and post clean latents
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)
            clean_latents_2x = current_history_2x
            clean_latents_4x = current_history_4x

            print(f"Clean latents prepared: Main shape {clean_latents.shape}, 2x shape {clean_latents_2x.shape}, 4x shape {clean_latents_4x.shape}")

            # --- Load Transformer for Sampling ---
            stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, f'Loading Transformer (Section {section_index})...'))))
            if not high_vram:
                # Unload VAE/ImageEncoder if they were loaded previously
                unload_complete_models(vae, image_encoder)
                # Move transformer, respecting memory preservation
                move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)
            else:
                 transformer.to(gpu) # Ensure it's on GPU

            # Configure TeaCache
            if use_teacache:
                transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
                print("TeaCache Enabled.")
            else:
                transformer.initialize_teacache(enable_teacache=False)
                print("TeaCache Disabled.")

            # --- Sampling Callback ---
            def callback(d):
                """Updates progress bar and preview during sampling."""
                if stream.input_queue.top() == 'end':
                    stream.output_queue.push(('end', None)) # Signal end
                    raise KeyboardInterrupt('User requested stop during sampling.')

                current_step = d['i'] + 1
                percentage = int(100.0 * current_step / steps)
                hint = f'Sampling Step {current_step}/{steps}'
                # Calculate current estimated video length for display
                current_total_frames = max(0, total_generated_latent_frames * 4 - 3) + (d['denoised'].shape[2] if d['denoised'] is not None else 0)
                current_video_seconds = max(0, current_total_frames / 30.0)
                desc = f'Section {section_index}/{len(latent_paddings)}. Est. Length: {current_video_seconds:.2f}s ({current_total_frames} frames).'

                preview = None
                if d.get('denoised') is not None:
                    try:
                        preview = vae_decode_fake(d['denoised']) # Use fake decode for speed
                        preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                        preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')
                    except Exception as e:
                         print(f"Warning: Preview generation failed - {e}")
                         preview = None # Prevent crash if preview fails

                stream.output_queue.push(('progress', (preview, desc, make_progress_bar_html(percentage, hint))))
                return

            # --- Perform Sampling ---
            print(f"Starting sampling for {num_frames_in_window} frames...")
            stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, f'Sampling Section {section_index}...'))))

            # Move conditioning tensors to GPU right before sampling
            device_kwargs = {'device': gpu, 'dtype': torch.bfloat16} # Transformer dtype
            text_kwargs = {'device': gpu, 'dtype': transformer.dtype} # Text embeds dtype
            latent_kwargs = {'device': gpu, 'dtype': torch.bfloat16} # Latents for sampling

            generated_latents = sample_hunyuan(
                transformer=transformer,
                sampler='unipc',
                width=width,
                height=height,
                frames=num_frames_in_window,
                real_guidance_scale=cfg,
                distilled_guidance_scale=gs,
                guidance_rescale=rs,
                num_inference_steps=steps,
                generator=rnd,
                prompt_embeds=llama_vec.to(**text_kwargs),
                prompt_embeds_mask=llama_attention_mask.to(gpu), # Mask is bool/int
                prompt_poolers=clip_l_pooler.to(**text_kwargs),
                negative_prompt_embeds=llama_vec_n.to(**text_kwargs),
                negative_prompt_embeds_mask=llama_attention_mask_n.to(gpu),
                negative_prompt_poolers=clip_l_pooler_n.to(**text_kwargs),
                device=gpu, # Target device for the sampler logic itself
                dtype=torch.bfloat16, # Sampling dtype
                image_embeddings=image_encoder_last_hidden_state.to(**text_kwargs),
                latent_indices=latent_indices.to(gpu),
                clean_latents=clean_latents.to(**latent_kwargs),
                clean_latent_indices=clean_latent_indices.to(gpu),
                clean_latents_2x=clean_latents_2x.to(**latent_kwargs),
                clean_latent_2x_indices=clean_latent_2x_indices.to(gpu),
                clean_latents_4x=clean_latents_4x.to(**latent_kwargs),
                clean_latent_4x_indices=clean_latent_4x_indices.to(gpu),
                callback=callback,
            )

            # Move generated latents back to CPU immediately after sampling
            generated_latents = generated_latents.to(device=cpu, dtype=torch.float32)
            print(f"Sampling complete for section {section_index}. Latent shape: {generated_latents.shape}")

            # Prepend start latent if it's the last section (which generates the beginning of the video)
            if is_last_section:
                print("Prepending start latent to the final generated segment.")
                generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)

            # Update history latents (on CPU)
            # Keep history size manageable if needed, but simple concat is okay for moderate lengths
            history_latents = torch.cat([generated_latents, history_latents], dim=2)
            total_generated_latent_frames += generated_latents.shape[2]
            print(f"History updated. Total latent frames: {total_generated_latent_frames}. History shape: {history_latents.shape}")


            # --- VAE Decoding Section ---
            stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, f'Decoding Section {section_index}...'))))
            if not high_vram:
                # Offload transformer, load VAE
                offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=8) # Preserve more if needed
                load_model_as_complete(vae, target_device=gpu, unload=False) # Don't unload others, just load VAE
            else:
                 vae.to(gpu) # Ensure VAE is on GPU

            # Determine which part of the history needs decoding for proper blending
            if is_first_section:
                # For first section, decode everything (it's small anyway)
                latents_to_decode = generated_latents
            else:
                # For subsequent sections, include enough context for proper blending
                overlap_frames = latent_window_size  # Number of frames needed for context from previous section
                # Get the appropriate slice from history_latents that includes the new frames plus overlap
                # The history now contains [new_latents, old_latents_from_previous_steps...]
                # We need new_latents + overlap_frames from the old_latents part
                start_idx_for_overlap = generated_latents.shape[2]
                end_idx_for_overlap = start_idx_for_overlap + overlap_frames
                # Ensure we don't index beyond the history bounds
                end_idx_for_overlap = min(end_idx_for_overlap, history_latents.shape[2])
                latents_to_decode = history_latents[:, :, :end_idx_for_overlap, :, :]

            print(f"Decoding latents of shape: {latents_to_decode.shape} (includes context for blending)")

            # Perform VAE decoding (potentially chunked if needed, but try full first)
            # Move latents to GPU for decoding
            current_pixels_section = vae_decode(latents_to_decode.to(gpu, vae.dtype), vae)
            current_pixels_section = current_pixels_section.cpu() # Move pixels back to CPU
            print(f"VAE decoding complete for section {section_index}. Pixel shape: {current_pixels_section.shape}")


            # --- Append Pixels ---
            if history_pixels is None:
                history_pixels = current_pixels_section
                print("Initialized pixel history.")
            else:
                # The number of overlapping *pixel* frames expected from the VAE decode
                # This overlap corresponds to the latent_window_size context used
                append_overlap_pixels = latent_window_size * 4

                # Calculate how many *new* pixel frames were actually generated in this VAE decode pass.
                # This corresponds to the 'generated_latents' part of 'latents_to_decode'.
                new_pixel_frames_in_section = generated_latents.shape[2] * 4

                # Ensure the calculated overlap doesn't exceed available frames
                actual_overlap = min(append_overlap_pixels, current_pixels_section.shape[2] - new_pixel_frames_in_section)
                actual_overlap = max(0, actual_overlap) # Ensure overlap is not negative

                print(f"Soft appending with pixel overlap: {actual_overlap} frames (Theoretical max: {append_overlap_pixels})")

                # Append with proper overlap
                # soft_append_bcthw expects (new_chunk, existing_history, overlap_count)
                history_pixels = soft_append_bcthw(current_pixels_section, history_pixels, actual_overlap)
                print("Pixel history appended.")


            # --- Offload VAE if Low VRAM ---
            if not high_vram:
                vae.to(cpu) # Offload VAE after use
                torch.cuda.empty_cache()
                print("VAE offloaded.")

            # --- Save Intermediate/Final Video ---
            # Save the current full video history
            output_filename = os.path.join(outputs_folder, f'{job_id}_{history_pixels.shape[2]}_frames.mp4') # Use history_pixels frame count
            final_num_frames = history_pixels.shape[2]
            print(f"Saving video: {output_filename} ({final_num_frames} frames)")
            save_bcthw_as_mp4(history_pixels, output_filename, fps=30, crf=mp4_crf)

            stream.output_queue.push(('file', output_filename)) # Update UI with latest video file

            # --- Delete Previous Intermediate Video if requested ---
            if keep_only_final_video and previous_output_filename is not None:
                try:
                    os.remove(previous_output_filename)
                    print(f"Deleted previous intermediate video: {previous_output_filename}")
                except OSError as e:
                    print(f"Warning: Could not delete intermediate video {previous_output_filename}: {e}")

            # Update the previous filename tracker *after* potential deletion
            previous_output_filename = output_filename

            if is_last_section:
                print("\n--- Reached last section. Generation finished. ---")
                break # Exit loop after the last section
            # --- End of Section Loop ---

    except KeyboardInterrupt:
         print("\nOperation cancelled by user during execution.")
         # No error traceback needed, finally block will handle cleanup
    except Exception as e:
        print("\n--- ERROR DURING GENERATION ---")
        traceback.print_exc()
        # Ensure UI knows an error occurred
        stream.output_queue.push(('progress', (None, f'Error: {type(e).__name__}', make_progress_bar_html(100, 'Error! Check Logs.'))))
    finally:
        # This block runs ALWAYS: after success, error, or KeyboardInterrupt
        print("\n--- Running Final Cleanup ---")
        # Send the final filename (or None if error occurred before first save)
        stream.output_queue.push(('end', output_filename))

        if not keep_models_alive:
            print("Unloading models as 'Keep Models in Memory' is unchecked.")
            # Use the specific unload function for all core models
            unload_all_models(all_core_models)
        else:
            print("Keeping models in memory as requested.")
            # In high VRAM mode, ensure models are back on GPU if they somehow got moved
            if high_vram:
                 print("High VRAM mode: Ensuring models are on GPU post-run...")
                 models_on_gpu = 0
                 for m in all_core_models:
                     try:
                         # Check parameter device first to avoid unnecessary moves
                         p = next(m.parameters(), None)
                         if p is not None and p.device != gpu:
                             m.to(gpu)
                             print(f"Moved {m.__class__.__name__} back to GPU.")
                         else:
                            # print(f"{m.__class__.__name__} already on GPU.") # Optional verbose log
                            pass
                         models_on_gpu += 1
                     except Exception as e:
                         print(f"Warning: Could not ensure {m.__class__.__name__} is on GPU: {e}")
                 if models_on_gpu == len(all_core_models):
                     print("All models confirmed/moved to GPU.")
                 torch.cuda.empty_cache()
            else:
                 print("Low VRAM mode: Models managed by DynamicSwap or loaded/unloaded as needed.")

        # Clean up DynamicSwap if it was installed during this run
        if dynamic_swap_installed and not high_vram:
             print("Uninstalling DynamicSwap...")
             try:
                 DynamicSwapInstaller.uninstall_model(transformer)
                 DynamicSwapInstaller.uninstall_model(text_encoder)
                 print("DynamicSwap uninstalled.")
             except Exception as e:
                 print(f"Warning: Error uninstalling DynamicSwap - {e}")

        print("Cleanup finished.")

    return # Worker function implicitly returns None


# --- Seed Button Functions ---
def randomize_seed_internal():
    """Internal function to just get a random seed value."""
    return random.randint(0, MAX_SEED)

# Define CSS classes for button states
base_button_class = "seed-button"
active_button_class = "seed-button-active"
inactive_button_class = "seed-button-inactive" # Optional: for specific inactive styling

def set_seed_mode_random():
    """Sets mode to random, updates button styles, and generates/sets new seed."""
    print("Seed mode set to RANDOM")
    new_seed = randomize_seed_internal()
    print(f"Generated and set random seed: {new_seed}")
    # Returns updates for the mode state, both buttons' classes, and the seed input
    return {
        seed_mode_state: SEED_MODE_RANDOM,
        random_seed_button: gr.update(elem_classes=[base_button_class, active_button_class]),
        use_last_seed_button: gr.update(elem_classes=[base_button_class]), # Reset to base/inactive
        seed: new_seed # Update seed field immediately
    }

def set_seed_mode_last(last_seed_value):
    """Sets mode to last, updates button styles, and sets seed input."""
    print("Seed mode set to LAST")
    seed_update = gr.update() # Default: no change
    if last_seed_value is not None and last_seed_value != -1:
        print(f"Setting seed input to last seed: {last_seed_value}")
        seed_update = gr.update(value=last_seed_value)
    else:
        print("No valid last seed available to set.")

    # Returns updates for mode state, both buttons' classes, and potentially the seed input
    return {
        seed_mode_state: SEED_MODE_LAST,
        random_seed_button: gr.update(elem_classes=[base_button_class]), # Reset to base/inactive
        use_last_seed_button: gr.update(elem_classes=[base_button_class, active_button_class]),
        seed: seed_update # Update seed field
    }
# --- End Seed Button Functions ---


def process(
    # Standard inputs
    input_image, end_image, prompt, n_prompt, seed_input, total_second_length, # Renamed seed -> seed_input
    latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation,
    use_teacache, mp4_crf, resolution, keep_models_alive, keep_only_final_video,
    # State inputs
    current_seed_mode, # Add seed mode state as input
):
    """Handles the Gradio button click, starts the worker, yields updates, determines and stores the seed."""
    global stream
    if input_image is None:
        gr.Warning("Please provide a Start Frame image.")
        # Return current state without starting
        return {
            result_video: None, preview_image: gr.update(visible=False),
            progress_desc: '', progress_bar: '',
            start_button: gr.update(interactive=True), end_button: gr.update(interactive=False),
            seed: gr.update(), last_seed_value: gr.update() # No changes
        }

    # --- Determine Seed to Use ---
    # Seed determination logic is simplified: we always use the value currently
    # in the seed_input field, as the random button now updates it directly.
    actual_seed = int(seed_input)
    print(f"--- Starting New Generation Request --- Seed: {actual_seed} (Mode: {current_seed_mode})")
    # --- End Determine Seed ---


    # Reset UI elements and update states
    yield {
        result_video: None, preview_image: gr.update(visible=False),
        progress_desc: '', progress_bar: '',
        start_button: gr.update(interactive=False), end_button: gr.update(interactive=True),
        # Update seed display *and* last seed state with the *actual* seed being used
        # No need to update seed display here if random button already did it.
        # We MUST update last_seed_value state though.
        seed: gr.update(value=actual_seed), # Ensure display matches seed used
        last_seed_value: actual_seed
    }


    stream = AsyncStream()

    # Pass the actual seed to the worker
    async_run(worker, input_image, end_image, prompt, n_prompt, actual_seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf, resolution, keep_models_alive, keep_only_final_video)

    output_filename = None # Keep track of the latest output file

    # Update loop for UI based on worker output
    while True:
        flag, data = stream.output_queue.next()

        if flag == 'file':
            output_filename = data
            yield {
                result_video: output_filename, preview_image: gr.update(visible=True),
                # Keep other things as they are during progress
                start_button: gr.update(interactive=False), end_button: gr.update(interactive=True)
            }
        elif flag == 'progress':
            preview, desc, html = data
            yield {
                result_video: output_filename, # Keep last known video
                preview_image: gr.update(visible=preview is not None, value=preview),
                progress_desc: desc, progress_bar: html,
                start_button: gr.update(interactive=False), end_button: gr.update(interactive=True)
            }
        elif flag == 'end':
             final_filename = data if data else output_filename
             print(f"Process received end signal. Final file: {final_filename}")
             yield {
                 result_video: final_filename, preview_image: gr.update(visible=False),
                 progress_desc: '', progress_bar: '',
                 start_button: gr.update(interactive=True), end_button: gr.update(interactive=False)
             }
             break
        else:
            print(f"Warning: Unknown flag received from worker: {flag}")


def end_process():
    """Sends the stop signal when the End Generation button is clicked."""
    print("End button clicked. Sending 'end' signal to worker.")
    if 'stream' in globals() and stream is not None:
        stream.input_queue.push('end')
    else:
        print("Warning: Stream object not found, cannot send end signal.")
    return gr.update(interactive=False)

# --- GPU Monitor Function ---
def get_gpu_stats_text():
    """Fetches GPU stats using pynvml and formats them for stable display."""
    global gpu_name # Access the global variable
    if not nvml_initialized or gpu_handle is None:
        return f"GPU ({gpu_name}): N/A" # Use consistent prefix

    try:
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
        util_rates = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
        temp_gpu = pynvml.nvmlDeviceGetTemperature(gpu_handle, pynvml.NVML_TEMPERATURE_GPU)

        # Calculate percentage
        if mem_info.total > 0:
            mem_usage_percent = (mem_info.used / mem_info.total) * 100
        else:
            mem_usage_percent = 0

        # Format memory values
        used_mem_str = format_bytes(mem_info.used)
        total_mem_str = format_bytes(mem_info.total)

        # Format stats using f-strings for fixed-width alignment
        stats_str = (
            f"{gpu_name:<15} | "
            f"Mem: {mem_usage_percent:>5.1f}% ({used_mem_str:>8s} / {total_mem_str:<8s}) | "
            f"Util: {util_rates.gpu:>3d}% | "
            f"Temp: {temp_gpu:>3d}°C"
        )

        return stats_str
    except pynvml.NVMLError as error:
        print(f"Error fetching GPU stats: {error}")
        return f"GPU ({gpu_name}): Error fetching stats" # Consistent prefix
    except Exception as e:
        print(f"Unexpected error fetching GPU stats: {e}")
        return f"GPU ({gpu_name}): Error" # Consistent prefix

# --- Modified update_gpu_display for older Gradio ---
def update_gpu_display():
    """Generator function for older Gradio versions to periodically update GPU stats."""
    print("Starting GPU stats update loop...")
    while True: # Run indefinitely
        stats_text = get_gpu_stats_text()
        yield gr.update(value=stats_text) # Yield the update for the textbox
        time.sleep(2) # Wait for 2 seconds before the next update
# --- End GPU Monitor Function ---


# --- Gradio UI Definition ---

quick_prompts = [
    'The girl dances gracefully, with clear movements, full of charm.',
    'A character doing some simple body movements.',
    'Astronaut floating in space overlooking earth.',
    'Robot walking through a futuristic city street.',
    'A dog chasing its tail in a park.',
    'Time-lapse clouds moving across a blue sky.',
]
quick_prompts = [[x] for x in quick_prompts]


css = make_progress_bar_css() + f"""
/* Style the GPU stats text box */
#gpu-stats-display textarea {{
    text-align: left !important;
    font-family: 'Consolas', 'Monaco', 'monospace';
    font-weight: bold !important;
    font-size: 0.85em !important;
    line-height: 1.2 !important;
    min-height: 20px !important;
    padding: 3px 6px !important;
    color: #c0c0c0 !important;
    border: none !important;
    background: transparent !important;
    white-space: pre;
}}
/* Adjust spacing for header row items */
#header-row > .wrap {{
    gap: 5px !important;
    align-items: center !important;
}}
/* Base style seed buttons */
.{base_button_class} {{
    min-width: 30px !important;
    max-width: 50px !important;
    height: 30px !important;
    margin: 0 2px !important;
    padding: 0 5px !important;
    line-height: 0 !important;
    border: 1px solid #555 !important; /* Default border */
    background-color: #333 !important; /* Default background */
    color: #eee !important; /* Default icon color */
    transition: background-color 0.2s ease, border-color 0.2s ease; /* Smooth transition */
}}
/* Active style for seed buttons */
.{active_button_class} {{
    border: 1px solid #aef !important; /* Highlight border */
    background-color: #446 !important; /* Highlight background */
    color: #fff !important; /* Highlight icon color */
}}
"""
block = gr.Blocks(css=css, theme=gr.themes.Soft()).queue() # Added a theme

with block:
    # --- State ---
    last_seed_value = gr.State(value=DEFAULT_SEED) # Store last used seed, init with default
    seed_mode_state = gr.State(value=SEED_MODE_LAST) # Track mode, default to 'last'

    gr.Markdown("<h1><center>FramePack I2V Video Generation</center></h1>")

    # --- Header Row with Description and GPU Stats ---
    with gr.Row(elem_id="header-row"):
        with gr.Column(scale=5):
            gr.Markdown("Generate video by extending frames based on a start image (and optional end image) and a prompt.")
        with gr.Column(scale=4, min_width=400):
            gpu_stats_display = gr.Textbox(
                value=f"GPU ({gpu_name}): Initializing...", label="GPU Stats", show_label=False,
                interactive=False, elem_id="gpu-stats-display"
            )
    # --- End Header Row ---

    with gr.Row():
        # Input Column
        with gr.Column(scale=1):
            gr.Markdown("## Inputs")
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(sources=['upload'], type="numpy", label="Start Frame", height=320)
                with gr.Column():
                    end_image = gr.Image(sources=['upload'], type="numpy", label="End Frame (Optional - for loops/transitions)", height=320)

            resolution = gr.Slider(label="Output Resolution (Width)", minimum=256, maximum=768, value=512, step=16, info="Nearest bucket (~WxH) will be used. Height adjusted automatically.")
            prompt = gr.Textbox(label="Prompt", placeholder="Describe the desired action or scene...", lines=2)
            example_quick_prompts = gr.Dataset(samples=quick_prompts, label='Prompt Examples', samples_per_page=len(quick_prompts), components=[prompt])
            example_quick_prompts.click(lambda x: x[0], inputs=[example_quick_prompts], outputs=prompt, show_progress=False, queue=False)

            with gr.Row():
                start_button = gr.Button("Generate Video", variant="primary", scale=3)
                end_button = gr.Button("Stop Generation", interactive=False, scale=1)

            with gr.Accordion("Advanced Settings", open=True):
                keep_models_alive = gr.Checkbox(label="Keep Models in GPU Memory After Generation", value=True, info="Recommended for faster subsequent runs, especially with high VRAM. Uncheck to free up VRAM afterwards.")
                keep_only_final_video = gr.Checkbox(label="Keep Only Final Video", value=True, info="Deletes intermediate video files after merging sections, saving only the final output.")
                use_teacache = gr.Checkbox(label='Use TeaCache Optimization', value=True, info='Generally faster, but may slightly affect fine details like hands/fingers.')

                # --- Seed Row ---
                with gr.Row(equal_height=True):
                    seed = gr.Number(label="Seed", value=DEFAULT_SEED, precision=0, minimum=0, maximum=MAX_SEED, step=1, info="Seed value. Updated by 🎲 / ♻️ buttons.", scale=4) # Updated info
                    # Set initial classes based on default mode (last)
                    random_seed_button = gr.Button("🎲", scale=1, elem_classes=[base_button_class])
                    use_last_seed_button = gr.Button("♻️", scale=1, elem_classes=[base_button_class, active_button_class]) # Active by default
                # --- End Seed Row ---

                total_second_length = gr.Slider(label="Target Video Length (Seconds)", minimum=1.0, maximum=120.0, value=5.0, step=0.1, info="Approximate desired duration.")
                steps = gr.Slider(label="Sampling Steps", minimum=10, maximum=60, value=25, step=1, info='Number of diffusion steps per frame window. Default (25) is recommended.')
                gs = gr.Slider(label="Guidance Scale (Distilled)", minimum=1.0, maximum=20.0, value=10.0, step=0.1, info='Strength of prompt guidance. Default (10) is recommended.')
                default_preserved_mem = max(4.0, free_mem_gb * 0.2) if not high_vram else 8.0
                gpu_memory_preservation = gr.Slider(label="Low VRAM: Min Free GPU Mem (GB)", minimum=1.0, maximum=20.0, value=default_preserved_mem, step=0.5, info="In Low VRAM mode, stops loading parts if free memory drops below this. Higher value = slower but safer. Less relevant in High VRAM mode.")
                mp4_crf = gr.Slider(label="MP4 Quality (CRF)", minimum=0, maximum=51, value=18, step=1, info="Constant Rate Factor for MP4 encoding. Lower = better quality & larger file (0=lossless, 18=high, 23=medium, 28=low).")

                # Hidden/Fixed parameters
                n_prompt = gr.Textbox(label="Negative Prompt", value="", visible=False)
                latent_window_size = gr.Slider(label="Latent Window Size", minimum=1, maximum=33, value=9, step=1, visible=False)
                cfg = gr.Slider(label="CFG Scale (Real)", minimum=1.0, maximum=32.0, value=1.0, step=0.01, visible=False)
                rs = gr.Slider(label="CFG Re-Scale", minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=False)

        # Output Column
        with gr.Column(scale=1):
            gr.Markdown("## Outputs")
            result_video = gr.Video(label="Generated Video", autoplay=True, show_share_button=False, height=512, loop=True, interactive=False, elem_id="result-video")
            preview_image = gr.Image(label="Sampling Preview", height=256, visible=False, interactive=False, elem_id="preview-image")
            progress_desc = gr.Markdown("", elem_classes='progress-text', elem_id="progress-desc")
            progress_bar = gr.HTML("", elem_classes='progress-bar-container', elem_id="progress-bar")
            gr.Markdown("ℹ️ **Note:** When using only a start frame, the model generates 'backwards' in time from the start frame during the first half of the process due to the sampling strategy. The final video plays forwards correctly. Using start and end frames guides the transition between them.")

    gr.HTML("""
    <div style="text-align:center; margin-top:20px; font-size:0.9em; color:#555;">
        Model: <a href="https://huggingface.co/lllyasviel/FramePackI2V_HY" target="_blank">FramePackI2V_HY</a> by lllyasviel |
        Based on <a href="https://huggingface.co/Tencent-Hunyuan/HunyuanDiT" target="_blank">Hunyuan-DiT</a> |
        UI built with Gradio
    </div>
    <div style="text-align:center; margin-top:5px; font-size:0.9em; color:#555;">
        Share your results and find ideas at the <a href="https://x.com/search?q=framepack&f=live" target="_blank">FramePack Twitter (X) thread</a>!
    </div>
    """)


    # --- Define Inputs/Outputs for process function ---
    process_inputs = [
        input_image, end_image, prompt, n_prompt, seed, total_second_length, # Pass seed input field
        latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation,
        use_teacache, mp4_crf, resolution, keep_models_alive, keep_only_final_video,
        seed_mode_state, # Pass the mode state
    ]
    process_outputs_dict = { # Use component references as keys
        result_video: None, preview_image: None, progress_desc: None, progress_bar: None,
        start_button: None, end_button: None,
        seed: None, # process function will update the seed display if needed
        last_seed_value: None # process function will update the last seed state
    }

    # --- Connect Buttons ---
    start_button.click(fn=process, inputs=process_inputs, outputs=list(process_outputs_dict.keys())) # Pass list of component references
    end_button.click(fn=end_process, inputs=None, outputs=[end_button])

    # --- Connect Seed Buttons ---
    # Random button click updates mode state, button classes, AND seed input field
    random_seed_button.click(
        fn=set_seed_mode_random,
        inputs=None,
        outputs=[seed_mode_state, random_seed_button, use_last_seed_button, seed] # ADDED seed to outputs
    )
    # Last seed button click updates mode state, button classes, and seed input field
    use_last_seed_button.click(
        fn=set_seed_mode_last,
        inputs=[last_seed_value], # Takes last seed state as input
        outputs=[seed_mode_state, random_seed_button, use_last_seed_button, seed] # Updates states, buttons, and seed field
    )

    # --- Schedule GPU Monitor Update (Old Gradio Workaround) ---
    if nvml_initialized:
        block.load(update_gpu_display, inputs=None, outputs=gpu_stats_display)


# Launch the Gradio App
block.launch(
    server_name=args.server,
    server_port=args.port,
    share=args.share,
    inbrowser=args.inbrowser,
)
