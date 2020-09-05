import torch
from pathlib import Path 

def save_checkpoint(generator, discriminator, optiG, optiD, epoch, errG, errD, run_note):
    '''
    Save G & D in the checkpoint repository
    '''
    Path("checkpoint/").mkdir(parents=True, exist_ok=True)

    torch.save(
        {
                'epoch': epoch,
                'model_state_dict': generator.state_dict(),
                'optimizer_state_dict': optiG.state_dict(),
                'loss': errG
        },
            "checkpoint/checkpointG-" + "DCGAN" + '-' + str(epoch) + '-' + str(round(errG.item(),2)) + '-' + run_note + '.pt'
    )
            
    torch.save(
        {
                'epoch': epoch,
                'model_state_dict': discriminator.state_dict(),
                'optimizer_state_dict': optiD.state_dict(),
                'loss': errD
        },
            'checkpoint/checkpointD-' + "DCGAN" + '-' + str(epoch) + '-' + str(round(errD.item(),2)) + '-' + run_note + '.pt'
    )