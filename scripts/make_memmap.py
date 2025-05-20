from argparse import ArgumentParser
from pathlib import Path

import tensordict as td
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


class ImageFolderWithPaths(datasets.ImageFolder):

    def __getitem__(self, index):

        img, label = super(ImageFolderWithPaths, self).__getitem__(index)

        path = self.imgs[index][0]

        return (img, label, path)


@torch.inference_mode()
def main():
    parser = ArgumentParser()
    parser.add_argument("--input_dir", type=Path)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--clip_features", type=Path)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    batch_size = args.batch_size
    clip_features_path = args.clip_features

    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = ImageFolderWithPaths(
        input_dir,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]),
    )

    clip_features_dict = torch.load(clip_features_path, weights_only=True)

    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        drop_last=False,
                        num_workers=16)

    image_data = []
    label_list = []
    all_clip_features = []
    for data in tqdm(loader):
        images, labels, paths = data
        # you need the clone to avoid out of memory error
        image_data.append(images.clone())
        label_list.append(labels.clone())

        paths = [Path(p) for p in paths]
        clip_features = [clip_features_dict[p.stem] for p in paths]
        all_clip_features.extend(clip_features)

    image_data = torch.cat(image_data, dim=0)
    label_list = torch.cat(label_list, dim=0)
    all_clip_features = torch.stack(all_clip_features, dim=0)

    td.TensorDict({
        'image': image_data,
        'label': label_list,
        'clip_features': all_clip_features,
    }).memmap_(output_dir)


if __name__ == '__main__':
    main()
