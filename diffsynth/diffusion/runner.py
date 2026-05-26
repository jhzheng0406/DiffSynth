import os, time, torch
from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    early_save_steps: int = 0,
    num_epochs: int = 1,
    args = None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        early_save_steps = getattr(args, 'early_save_steps', 0)
        num_epochs = args.num_epochs

    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    model.to(device=accelerator.device)
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    initialize_deepspeed_gradient_checkpointing(accelerator)

    step_offset = getattr(model_logger, "step_offset", 0)

    log_file = None
    if accelerator.is_main_process:
        os.makedirs(model_logger.output_path, exist_ok=True)
        log_file = open(os.path.join(model_logger.output_path, "train.log"), "a", buffering=1)
        offset_note = f" step_offset={step_offset}" if step_offset else ""
        log_file.write(f"=== run start {time.strftime('%Y-%m-%d %H:%M:%S')} | lr={learning_rate} epochs={num_epochs}{offset_note} ===\n")

    local_step = 0
    for epoch_id in range(num_epochs):
        pbar = tqdm(dataloader, disable=not accelerator.is_main_process)
        for data in pbar:
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                accelerator.backward(loss)
                optimizer.step()
                model_logger.on_step_end(accelerator, model, save_steps, early_save_steps=early_save_steps, loss=loss)
                scheduler.step()
                if accelerator.is_main_process:
                    local_step += 1
                    global_step = step_offset + local_step
                    loss_val = loss.detach().float().item()
                    pbar.set_postfix(loss=f"{loss_val:.4f}")
                    log_file.write(f"{time.strftime('%H:%M:%S')} epoch={epoch_id} step={global_step} loss={loss_val:.6f}\n")
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, save_steps)
    if log_file is not None:
        log_file.write(f"=== run end {time.strftime('%Y-%m-%d %H:%M:%S')} | total_steps={step_offset + local_step} ===\n")
        log_file.close()


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
):
    if args is not None:
        num_workers = args.dataset_num_workers
        
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    model.to(device=accelerator.device)
    model, dataloader = accelerator.prepare(model, dataloader)
    
    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data)
                torch.save(data, save_path)


def initialize_deepspeed_gradient_checkpointing(accelerator: Accelerator):
    if getattr(accelerator.state, "deepspeed_plugin", None) is not None:
        ds_config = accelerator.state.deepspeed_plugin.deepspeed_config
        if "activation_checkpointing" in ds_config:
            import deepspeed
            act_config = ds_config["activation_checkpointing"]
            deepspeed.checkpointing.configure(
                mpu_=None, 
                partition_activations=act_config.get("partition_activations", False),
                checkpoint_in_cpu=act_config.get("cpu_checkpointing", False),
                contiguous_checkpointing=act_config.get("contiguous_memory_optimization", False)
            )
        else:
            print("Do not find activation_checkpointing config in deepspeed config, skip initializing deepspeed gradient checkpointing.")
