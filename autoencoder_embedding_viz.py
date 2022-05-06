# %%
from torch.utils.tensorboard import SummaryWriter
from autoencoder import MyDataset, Mapper, DataLoader
from torchvision.transforms import ToTensor
from tqdm import tqdm
import torch
from tsnecuda import TSNE

# %%
writer = SummaryWriter(log_dir="/home/ubuntu/tensorboard/autoencoder")

# %%
autoencoder = Mapper.load_from_checkpoint(
    "outputs/bedroom_256_autoencoder/2022-04-05-10-42-05/ckpts/autoencoder-epoch58-step50000.ckpt"
).eval().to("cuda:0")
autoencoder.freeze()

# %%
dataset = DataLoader(
    MyDataset(root="/home/ubuntu/bedroom_256_train/", transform=ToTensor()),
    batch_size=16,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    drop_last=False,
)

# %%
label_img = []
embedding = []
for i, x in tqdm(enumerate(dataset)):
    if i > 50:
        break

    x = x.to("cuda:0")
    z, _ = autoencoder(x)

    x = (x + 1) * 0.5
    z = z.view(x.shape[0], -1)

    label_img.append(x.cpu())
    embedding.append(z.cpu())


# %%
embedding = torch.cat(embedding)
label_img = torch.cat(label_img)

# %%
projected_embedding = TSNE(verbose=1).fit_transform(embedding)

# %%
writer.add_embedding(
    mat=projected_embedding,
    label_img=label_img,
)

# %%
writer.close()
