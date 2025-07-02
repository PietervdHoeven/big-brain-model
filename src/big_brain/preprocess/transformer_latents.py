import torch
import argparse
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

from big_brain.models.autoencoders import AutoEncoder
from big_brain.models.encoders import Encoder
from big_brain.models.decoders import Decoder

Braincoder = NotImplemented # Encoder part of the AutoEncoder with trained weights (we have the weights saved as a checkpoint)

def main(in_dir, out_dir):
    # 1. setup encoder for compressing
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Init a clean autoencoder class
    ae = AutoEncoder(Encoder(), Decoder())
    # Load the checkpoint
    ckpt = torch.load("checkpoints/AutoEncoder_512_v1/checkpoint.pth")
    # Transfer the weights to the autoencoder
    ae.load_state_dict(ckpt["model"])
    # Only take the encoder part of the ae
    encoder = ae.encoder.to(device)
    encoder.eval()

    # 2. For each session, compress the normalised volumes and stack them into sequences of latents with accompanying gradient info
    for dirpath, _, filenames in tqdm(os.walk(in_dir)):
        zs = []
        gs = []
        for name in filenames:
            # Load .npz file
            file = os.path.join(dirpath, name)
            data = np.load(file)
            # Encode volume into a latent z
            x = torch.from_numpy(data["vol_data"]).float().unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                z = encoder(x)
            z = z.view(-1).cpu().numpy()
            # Get the gradient information
            g = np.append(data["bvec"], data["bval"])
            # Save all latents and gradients into list
            zs.append(z)
            gs.append(g)
        # Only save files when walk hit a dir with .npz files
        if len(zs) != 0:
            # Stack latents
            Z = np.stack(zs)    # [N, 512]
            G = np.stack(gs)    # [N, 4]
            # Get metadata
            p_id = data["patient"]
            s_id = data["session"]
            # Prepare output dir for this session
            ses_dir = out_dir / p_id / s_id
            ses_dir.mkdir(parents=True, exist_ok=True)
            out_file = ses_dir / f"{p_id}_{s_id}_latent.npz"
            # Save file
            np.savez_compressed(
                file=out_file,
                z = Z,
                g = G,
                patient = p_id,
                session = s_id
            )
            #print(f"Saved latents and gradients for session {p_id}_{s_id} to: {out_file}")

if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Compress dwi volumes into latents z paired with gradients g. Stack zs into Z and gs into G")
    p.add_argument(
        "--in_dir", type=Path,
        default=Path("/home/spieterman/dev/big-brain-model/data/encoder"),
        help="Where to find *_gradxxx.npz files."
    )
    p.add_argument(
        "--out_dir", type=Path,
        default=Path("/home/spieterman/dev/big-brain-model/data/transformer"),
        help="Where to write per-gradient .npz files."
    )

    args = p.parse_args()
    main(args.in_dir, args.out_dir)