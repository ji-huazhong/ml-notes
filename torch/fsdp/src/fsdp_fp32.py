"""
PyTorch FSDP (Fully Sharded Data Parallel) Training Demo

This script demonstrates how to use FSDP for distributed training.
FSDP shards model parameters, gradients, and optimizer states across GPUs,
enabling training of very large models that don't fit in a single GPU memory.

Usage:
    # Single GPU (for testing)
    python fsdp_fp32.py
    
    # Multi-GPU with torchrun
    torchrun --nproc_per_node=2 fsdp_fp32.py
    
    # Multi-node (example)
    torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr=192.168.1.1 fsdp_fp32.py
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from functools import partial
import argparse


class SimpleLinearModel(nn.Module):
    def __init__(self, input_size=512, hidden_size=512, output_size=10000):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.layer4 = nn.Linear(hidden_size, output_size)

        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        x = self.relu(x)
        x = self.layer4(x)
        return x


class SyntheticDataset(Dataset):
    """Synthetic dataset for demonstration"""
    def __init__(self, num_samples=10000, vocab_size=10000, hidden_size=512):
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 生成随机输入特征 (hidden_size,)
        inputs = torch.randn(self.hidden_size)
        # 生成标签：单个类别 (vocab_size个类别)
        labels = torch.randint(0, self.vocab_size, ()).long()
        return inputs, labels


def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        # Single GPU mode
        rank = 0
        world_size = 1
        local_rank = 0
    
    # Set device
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    # Initialize process group
    if world_size > 1:
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            rank=rank,
            world_size=world_size
        )
    
    return rank, world_size, local_rank, device


def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def build_fsdp_model(model, device, args):
    """Wrap model with FSDP"""
    # 使用size_based策略：当参数数量超过min_num_params时自动wrap
    auto_wrap_policy = partial(
        size_based_auto_wrap_policy,
        min_num_params=1000  # 可以根据需要调整
    )
    
    # Sharding strategy
    sharding_strategy = ShardingStrategy.FULL_SHARD  # Default: shard everything
    
    if args.sharding_strategy == 'full_shard':
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif args.sharding_strategy == 'shard_grad_op':
        sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
    elif args.sharding_strategy == 'no_shard':
        sharding_strategy = ShardingStrategy.NO_SHARD
    
    # Wrap model with FSDP (using FP32)
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=sharding_strategy,
        device_id=device,
    )
    
    return model


def train_step(model, inputs, labels, optimizer, loss_fn, device):
    """Train for one step (one batch)"""
    inputs = inputs.to(device)
    labels = labels.to(device)
    
    optimizer.zero_grad()
    outputs = model(inputs)  # (batch_size, vocab_size)
    loss = loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()
    
    return loss.item()


def main(args):
    # Default values
    vocab_size = 10000
    hidden_size = 512
    
    # Setup distributed training
    rank, world_size, local_rank, device = setup_distributed()
    
    if rank == 0:
        print(f'Starting FSDP training with {world_size} GPU(s)')
        print(f'Device: {device}')
        print(f'Sharding strategy: {args.sharding_strategy}')
        print(f'Precision: FP32')
        print(f'Model type: Simple 4-Layer Linear Model')
    
    # Create simple 4-layer Linear model
    model = SimpleLinearModel(
        input_size=hidden_size,  # 使用hidden_size作为输入维度
        hidden_size=hidden_size,
        output_size=vocab_size
    )
    
    # Move model to device and wrap with FSDP
    model = model.to(device)
    model = build_fsdp_model(model, device, args)
    
    # Print model info
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Total parameters: {total_params:,}')
        print(f'Trainable parameters: {trainable_params:,}')
        print(f'Model structure:')
        print(f'  Input size: {hidden_size}')
        print(f'  Hidden size: {hidden_size}')
        print(f'  Output size: {vocab_size}')
        print(f'  Layers: 4 Linear layers with ReLU activation')
    
    # Create dataset and dataloader
    train_dataset = SyntheticDataset(
        num_samples=args.num_samples,
        vocab_size=vocab_size,
        hidden_size=hidden_size
    )
    
    # Use DistributedSampler for multi-GPU training
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    ) if world_size > 1 else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=2,
        pin_memory=True
    )
    
    # Loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Training loop
    model.train()
    if rank == 0:
        print('\nStarting training...\n')
        print(f'Total steps: {args.total_steps}\n')
    
    global_step = 0
    train_iter = iter(train_loader)
    epoch_counter = 0
    
    while global_step < args.total_steps:
        try:
            inputs, labels = next(train_iter)
        except StopIteration:
            # Data exhausted, create new iterator
            epoch_counter += 1
            if train_sampler is not None:
                train_sampler.set_epoch(epoch_counter)
            train_iter = iter(train_loader)
            inputs, labels = next(train_iter)
        
        # Train step
        loss_value = train_step(model, inputs, labels, optimizer, loss_fn, device)
        
        global_step += 1
        
        # Logging
        if rank == 0:
            print(f'Step {global_step}/{args.total_steps}, Loss: {loss_value:.4f}')
    
    # Save model
    if args.save_model and rank == 0:
        save_path = 'fsdp_model_checkpoint.pt'
        # Note: For FSDP, we need to use state_dict with state_dict_type
        from torch.distributed.fsdp import StateDictType, FullStateDictConfig
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': args,
            }, save_path)
        print(f'Model saved to {save_path}')
    
    # Cleanup
    cleanup_distributed()
    
    if rank == 0:
        print('Training completed!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FSDP Training Demo')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size per GPU')
    parser.add_argument('--total_steps', type=int, default=100, help='Total number of training steps')
    parser.add_argument('--num_samples', type=int, default=256, help='Number of training samples')
    parser.add_argument('--sharding_strategy', type=str, default='full_shard',
                        choices=['full_shard', 'shard_grad_op', 'no_shard'],
                        help='FSDP sharding strategy')
    parser.add_argument('--save_model', action='store_true', help='Save model checkpoint')
    
    args = parser.parse_args()
    
    main(args)
