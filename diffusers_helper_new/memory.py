import torch


cpu = torch.device('cpu')
gpu = torch.device(f'cuda:{torch.cuda.current_device()}')
gpu_complete_modules = []


class DynamicSwapInstaller:
    # ... (rest of the class remains the same) ...
    @staticmethod
    def _install_module(module: torch.nn.Module, **kwargs):
        original_class = module.__class__
        module.__dict__['forge_backup_original_class'] = original_class

        def hacked_get_attr(self, name: str):
            if '_parameters' in self.__dict__:
                _parameters = self.__dict__['_parameters']
                if name in _parameters:
                    p = _parameters[name]
                    if p is None:
                        return None
                    if p.__class__ == torch.nn.Parameter:
                        return torch.nn.Parameter(p.to(**kwargs), requires_grad=p.requires_grad)
                    else:
                        return p.to(**kwargs)
            if '_buffers' in self.__dict__:
                _buffers = self.__dict__['_buffers']
                if name in _buffers:
                    return _buffers[name].to(**kwargs)
            # Fallback to avoid errors if super doesn't have __getattr__ or for special attributes
            try:
                return super(original_class, self).__getattr__(name)
            except AttributeError:
                 # Standard attribute access
                 return object.__getattribute__(self, name)


        module.__class__ = type('DynamicSwap_' + original_class.__name__, (original_class,), {
            '__getattr__': hacked_get_attr,
        })

        return

    @staticmethod
    def _uninstall_module(module: torch.nn.Module):
        if 'forge_backup_original_class' in module.__dict__:
            module.__class__ = module.__dict__.pop('forge_backup_original_class')
        return

    @staticmethod
    def install_model(model: torch.nn.Module, **kwargs):
        for m in model.modules():
            DynamicSwapInstaller._install_module(m, **kwargs)
        return

    @staticmethod
    def uninstall_model(model: torch.nn.Module):
        for m in model.modules():
            DynamicSwapInstaller._uninstall_module(m)
        return


def fake_diffusers_current_device(model: torch.nn.Module, target_device: torch.device):
    if hasattr(model, 'scale_shift_table'):
        model.scale_shift_table.data = model.scale_shift_table.data.to(target_device)
        return

    for k, p in model.named_modules():
        if hasattr(p, 'weight'):
            p.to(target_device)
            return


def get_cuda_free_memory_gb(device=None):
    if device is None:
        device = gpu

    memory_stats = torch.cuda.memory_stats(device)
    bytes_active = memory_stats['active_bytes.all.current']
    bytes_reserved = memory_stats['reserved_bytes.all.current']
    bytes_free_cuda, _ = torch.cuda.mem_get_info(device)
    bytes_inactive_reserved = bytes_reserved - bytes_active
    bytes_total_available = bytes_free_cuda + bytes_inactive_reserved
    return bytes_total_available / (1024 ** 3)


def move_model_to_device_with_memory_preservation(model, target_device, preserved_memory_gb=0):
    print(f'Moving {model.__class__.__name__} to {target_device} with preserved memory: {preserved_memory_gb} GB')

    for m in model.modules():
        if get_cuda_free_memory_gb(target_device) <= preserved_memory_gb:
            torch.cuda.empty_cache()
            return # Stop moving if memory limit reached

        # Check if the module has parameters or buffers directly attached
        # (e.g., Linear, Conv2d) before trying to move its weight.
        # This avoids trying to move weights of container modules.
        if hasattr(m, '_parameters') and m._parameters:
            m.to(device=target_device)
        elif hasattr(m, '_buffers') and m._buffers:
             m.to(device=target_device)
        # Fallback for modules that might store weight differently but are movable
        elif hasattr(m, 'weight') and isinstance(m.weight, torch.Tensor):
            m.to(device=target_device)


    # Ensure the top-level model device attribute is set
    model.to(device=target_device)
    torch.cuda.empty_cache()
    return


def offload_model_from_device_for_memory_preservation(model, target_device, preserved_memory_gb=0):
    print(f'Offloading {model.__class__.__name__} from {target_device} to preserve memory: {preserved_memory_gb} GB')

    # Offload modules with parameters/buffers first
    for m in reversed(list(model.modules())): # Offload leaf modules first potentially
        if get_cuda_free_memory_gb(target_device) >= preserved_memory_gb:
            torch.cuda.empty_cache()
            return # Stop offloading if memory goal reached

        if hasattr(m, '_parameters') and m._parameters:
            m.to(device=cpu)
        elif hasattr(m, '_buffers') and m._buffers:
            m.to(device=cpu)
        elif hasattr(m, 'weight') and isinstance(m.weight, torch.Tensor):
             m.to(device=cpu)

    # Ensure the top-level model device attribute is set
    model.to(device=cpu)
    torch.cuda.empty_cache()
    return


def unload_complete_models(*args):
    """Unloads models currently marked as 'complete' on GPU and any additional models passed."""
    unloaded_count = 0
    models_to_unload = list(set(gpu_complete_modules + list(args))) # Combine and unique
    for m in models_to_unload:
        if hasattr(m, 'to'):
            try:
                m.to(device=cpu)
                print(f'Unloaded {m.__class__.__name__} to CPU.')
                unloaded_count += 1
            except Exception as e:
                print(f"Could not unload {m.__class__.__name__}: {e}")

    gpu_complete_modules.clear()
    if unloaded_count > 0:
        torch.cuda.empty_cache()
    print(f"Unloaded {unloaded_count} models.")
    return

# NEW function specifically for unloading all core models at the end
def unload_all_models(models_list):
    """Unloads all models in the provided list to CPU."""
    unloaded_count = 0
    print("Attempting to unload all core models to CPU...")
    for m in models_list:
         if hasattr(m, 'to'):
            try:
                m.to(device=cpu)
                print(f'Unloaded {m.__class__.__name__} to CPU.')
                unloaded_count += 1
            except Exception as e:
                print(f"Could not unload {m.__class__.__name__}: {e}")

    gpu_complete_modules.clear() # Clear this list as well for consistency
    if unloaded_count > 0:
        torch.cuda.empty_cache()
    print(f"Unloaded {unloaded_count} core models.")
    return


def load_model_as_complete(model, target_device, unload=True):
    if unload:
        unload_complete_models() # Unload previous 'complete' models

    # Check if model is already on the target device
    try:
        # A simple check using a parameter or buffer
        p = next(model.parameters(), None)
        if p is not None and p.device == target_device:
             print(f'{model.__class__.__name__} is already on {target_device}.')
             if model not in gpu_complete_modules:
                 gpu_complete_modules.append(model)
             return
        b = next(model.buffers(), None)
        if b is not None and b.device == target_device:
             print(f'{model.__class__.__name__} is already on {target_device}.')
             if model not in gpu_complete_modules:
                 gpu_complete_modules.append(model)
             return
    except Exception:
        pass # Fallback to moving if check fails

    model.to(device=target_device)
    print(f'Loaded {model.__class__.__name__} to {target_device} as complete.')

    if model not in gpu_complete_modules:
        gpu_complete_modules.append(model)
    return