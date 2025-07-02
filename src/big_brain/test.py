from big_brain.models.transformer import DWIBert

dwibert = DWIBert(
    d_model=384,
    n_head=6,
    d_ff=1536,
    n_layers=6,
    dropout=0.1,
    lr=1e-4,
    weight_decay=1e-2
)

print(dwibert)