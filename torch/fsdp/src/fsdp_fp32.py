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

from model import SimpleLinearModel
from profiler import MemoryProfiler, null_context


class SyntheticDataset(Dataset):
    """Synthetic dataset for demonstration"""
    def __init__(self, num_samples=10000, vocab_size=10000, seq_len=128, hidden_size=512):
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 生成随机输入特征 (seq_len, hidden_size)
        # 使用正态分布生成，模拟embedding后的特征
        inputs = torch.randn(self.seq_len, self.hidden_size)
        # 生成标签：每个位置对应一个类别 (vocab_size个类别)
        labels = torch.randint(0, self.vocab_size, (self.seq_len,))
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


def get_fsdp_model(model, device, args):
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


def train_step(model, inputs, labels, optimizer, criterion, device):
    """Train for one step (one batch)"""
    inputs = inputs.to(device)
    labels = labels.to(device)
    
    optimizer.zero_grad()
    outputs = model(inputs)  # (batch_size, seq_len, vocab_size)
    # Reshape for CrossEntropyLoss: (batch_size * seq_len, vocab_size) and (batch_size * seq_len,)
    outputs = outputs.reshape(-1, outputs.size(-1))
    labels = labels.reshape(-1)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    
    return loss.item()


def main(args):
    
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
        input_size=args.hidden_size,  # 使用hidden_size作为输入维度
        hidden_size=args.hidden_size,
        output_size=args.vocab_size
    )
    
    # Move model to device and wrap with FSDP
    model = model.to(device)
    model = get_fsdp_model(model, device, args)
    
    # Print model info
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Total parameters: {total_params:,}')
        print(f'Trainable parameters: {trainable_params:,}')
        print(f'Model structure:')
        print(f'  Input size: {args.hidden_size}')
        print(f'  Hidden size: {args.hidden_size}')
        print(f'  Output size: {args.vocab_size}')
        print(f'  Layers: 4 Linear layers with ReLU activation')
    
    # Create dataset and dataloader
    train_dataset = SyntheticDataset(
        num_samples=args.num_samples,
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        hidden_size=args.hidden_size  # 添加hidden_size参数
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
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # Initialize memory profiler if enabled
    profiler = None
    if args.enable_memory_profiling:
        profiler = MemoryProfiler(
            output_dir=args.memory_profiling_dir,
            enabled=True,
            rank=rank,
            enabled_ranks=args.memory_profiling_ranks,
            dump_on_enter=False,
            dump_on_exit=True
        )
        profiler.start_recording()
    
    # Training loop
    model.train()
    if rank == 0:
        print('\nStarting training...\n')
    
    global_step = 0
    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        if rank == 0:
            print(f'Epoch {epoch + 1}/{args.epochs}')
        
        epoch_loss = 0.0
        num_batches = 0
        
        # Profile epoch if enabled
        epoch_ctx = profiler.profile(step=None) if profiler else null_context()
        with epoch_ctx:
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                # Set current step for profiler (useful for decorator mode)
                if profiler is not None:
                    profiler.set_step(global_step)
                
                # Profile batch if enabled (only at log intervals to avoid too many snapshots)
                batch_ctx = (profiler.profile(step=global_step) 
                            if (profiler and batch_idx % args.log_interval == 0) 
                            else null_context())
                
                with batch_ctx:
                    # Train step
                    loss_value = train_step(model, inputs, labels, optimizer, criterion, device)
                    
                    epoch_loss += loss_value
                    num_batches += 1
                    global_step += 1
                    
                    # Logging
                    if rank == 0 and batch_idx % args.log_interval == 0:
                        print(f'Step {global_step}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss_value:.4f}')
            
            # Update epoch profile with final step (for exit dump)
            if profiler is not None:
                # Use the last completed step (global_step - 1) for epoch exit dump
                profiler.set_step(global_step - 1 if global_step > 0 else 0)
        
        # Epoch summary
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        if rank == 0:
            print(f'Epoch {epoch + 1} - Train Loss: {avg_loss:.4f}\n')
    
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
    
    # Stop profiler
    if profiler is not None:
        # Use the last step for final snapshot
        final_step = global_step - 1 if global_step > 0 else 0
        profiler.set_step(final_step)
        profiler.dump_snapshot(step=final_step)
        profiler.stop_recording()
        if rank == 0:
            print(f'\n[MemoryProfiler] All snapshots saved to: {profiler.output_dir}')
            print('[MemoryProfiler] To visualize snapshots, use torch.cuda.memory._dump_snapshot() or other visualization tools')
    
    # Cleanup
    cleanup_distributed()
    
    if rank == 0:
        print('Training completed!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FSDP Training Demo')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--vocab_size', type=int, default=10000, help='Output vocabulary size')
    parser.add_argument('--hidden_size', type=int, default=512, help='Hidden size')
    parser.add_argument('--seq_len', type=int, default=128, help='Sequence length')
    parser.add_argument('--num_samples', type=int, default=256, help='Number of training samples')
    parser.add_argument('--sharding_strategy', type=str, default='full_shard',
                        choices=['full_shard', 'shard_grad_op', 'no_shard'],
                        help='FSDP sharding strategy')
    parser.add_argument('--log_interval', type=int, default=1, help='Log interval')
    parser.add_argument('--save_model', action='store_true', help='Save model checkpoint')
    
    # Memory profiling arguments
    parser.add_argument('--enable_memory_profiling', action='store_true',
                        help='Enable CUDA memory profiling with history recording and snapshots')
    parser.add_argument('--memory_profiling_dir', type=str, default='memory_snapshots',
                        help='Directory to save memory snapshots')
    parser.add_argument('--memory_profiling_ranks', type=str, default='0',
                        help='Comma-separated list of ranks to dump snapshots (e.g., "0,1,2" or "0"). Default: "0"')
    
    args = parser.parse_args()
    
    # Parse memory_profiling_ranks from string to list of integers
    if args.enable_memory_profiling:
        try:
            args.memory_profiling_ranks = [int(r.strip()) for r in args.memory_profiling_ranks.split(',')]
        except ValueError:
            raise ValueError(f'Invalid memory_profiling_ranks format: {args.memory_profiling_ranks}. '
                           f'Expected comma-separated integers (e.g., "0,1,2").')

    main(args)
