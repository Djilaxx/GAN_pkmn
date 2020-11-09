import inspect, sys, os, re
import torch
from pathlib import Path 

def save_checkpoint(generator, discriminator, optiG, optiD, epoch, loss_fct, errG, errD, run_note):
    '''
    Save G & D in the checkpoint repository
    '''
    Path("checkpoint/").mkdir(parents=True, exist_ok=True)
    loss_file = os.path.basename(inspect.getmodule(loss_fct.__class__).__file__).replace('.py', '')
    G_file = os.path.basename(inspect.getmodule(generator.__class__).__file__).replace('.py', '')
    G_folder = os.path.basename(os.path.dirname(inspect.getmodule(generator.__class__).__file__))
    D_file = os.path.basename(inspect.getmodule(discriminator.__class__).__file__).replace('.py', '')
    D_folder = os.path.basename(os.path.dirname(inspect.getmodule(discriminator.__class__).__file__))

    torch.save(
        {
                'epoch': epoch,
                'model_state_dict': generator.state_dict(),
                'optimizer_state_dict': optiG.state_dict(),
                'loss': errG
        },
        f"checkpoint/checkpointG-{G_folder}_{G_file}_{loss_file}_{epoch}_{round(errG.item(),2)}_{run_note}.pt"
    )
            
    torch.save(
        {
                'epoch': epoch,
                'model_state_dict': discriminator.state_dict(),
                'optimizer_state_dict': optiD.state_dict(),
                'loss': errD
        },
        f"checkpoint/checkpointD-{D_folder}_{D_file}_{loss_file}_{epoch}_{round(errD.item(),2)}_{run_note}.pt"
    )