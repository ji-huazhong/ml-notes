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
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from functools import partial
import argparse

from model import TransformerBlock, DemoModel


class SyntheticDataset(Dataset):
    """Synthetic dataset for demonstration"""
    def __init__(self, num_samples=10000, vocab_size=10000, seq_len=128):
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random input sequence
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        # Generate labels for each position (for language modeling, label is next token)
        labels = torch.randint(0, self.vocab_size, (self.seq_len,))
        return input_ids, labels


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
    # Auto-wrap policy for transformer blocks
    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock}
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


def train_epoch(model, dataloader, optimizer, criterion, device, rank, args):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (inputs, labels) in enumerate(dataloader):
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
        
        total_loss += loss.item()
        num_batches += 1
        
        if rank == 0 and batch_idx % args.log_interval == 0:
            print(f'Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}')
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def main(args):
    
    # Setup distributed training
    rank, world_size, local_rank, device = setup_distributed()
    
    if rank == 0:
        print(f'Starting FSDP training with {world_size} GPU(s)')
        print(f'Device: {device}')
        print(f'Sharding strategy: {args.sharding_strategy}')
        print(f'Precision: FP32')
    
    # Create model
    model = DemoModel(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        intermediate_size=args.intermediate_size,
        max_seq_len=args.seq_len
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
    
    # Create dataset and dataloader
    train_dataset = SyntheticDataset(
        num_samples=args.num_samples,
        vocab_size=args.vocab_size,
        seq_len=args.seq_len
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
    
    # Training loop
    if rank == 0:
        print('\nStarting training...\n')
    
    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        if rank == 0:
            print(f'Epoch {epoch + 1}/{args.epochs}')
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, rank, args)
        
        if rank == 0:
            print(f'Epoch {epoch + 1} - Train Loss: {train_loss:.4f}\n')
    
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
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--vocab_size', type=int, default=10000, help='Vocabulary size')
    parser.add_argument('--hidden_size', type=int, default=512, help='Hidden size')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of transformer layers')
    parser.add_argument('--intermediate_size', type=int, default=2048, help='Intermediate size')
    parser.add_argument('--seq_len', type=int, default=128, help='Sequence length')
    parser.add_argument('--num_samples', type=int, default=512, help='Number of training samples')
    parser.add_argument('--sharding_strategy', type=str, default='full_shard',
                        choices=['full_shard', 'shard_grad_op', 'no_shard'],
                        help='FSDP sharding strategy')
    parser.add_argument('--log_interval', type=int, default=1, help='Log interval')
    parser.add_argument('--save_model', action='store_true', help='Save model checkpoint')
    
    args = parser.parse_args()

    main(args)
