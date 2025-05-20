import argparse
import json
import os
from pathlib import Path

import open_clip
import torch
from open_clip import create_model_from_pretrained
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--imagenet_path",
        type=str,
        required=True,
        help="Path to the ImageNet directory (e.g., ../data/imagenet/train_blurred_32/box)",
    )
    parser.add_argument(
        "--caption_path",
        type=str,
        required=True,
        help=
        "Path to the JSON file with captions (e.g., ../data/imagenet/imagenet-1k-vl-enriched/train_captions.json)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the output features (e.g., ./data/imagenet/train_clip_captions.pth)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
    )
    args = parser.parse_args()

    imagenet_path = Path(args.imagenet_path).expanduser()
    caption_path = Path(args.caption_path).expanduser()
    output_path = Path(args.output_path).expanduser()
    batch_size = args.batch_size

    # Read JSON captions
    with open(caption_path, 'r') as f:
        captions = json.load(f)

    all_image_ids = []
    classes = sorted(os.listdir(imagenet_path))
    for this_class in classes:
        images = sorted(os.listdir(imagenet_path / this_class))
        images = [i[:-4] for i in images]  # remove file extensions
        all_image_ids.extend(images)

    all_image_ids_from_caption = set(captions.keys())
    all_image_ids = set(all_image_ids)

    print(f'Number of keys in caption: {len(all_image_ids_from_caption)}')
    print(f'Number of keys in image: {len(all_image_ids)}')
    print(f'Number of keys not found in caption: {len(all_image_ids - all_image_ids_from_caption)}')
    print(f'Number of keys not found in image: {len(all_image_ids_from_caption - all_image_ids)}')

    model, _ = create_model_from_pretrained('hf-hub:apple/DFN5B-CLIP-ViT-H-14')
    model = model.cuda().eval()
    tokenizer = open_clip.get_tokenizer('ViT-H-14')

    model.encode_text = torch.compile(model.encode_text)

    with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        output_feature = {}

        all_image_ids_list = list(all_image_ids)
        for i in tqdm(range(0, len(all_image_ids_list), batch_size)):
            batch_image_ids = all_image_ids_list[i:i + batch_size]
            batch_captions = [captions[i] for i in batch_image_ids]

            tokens = tokenizer(batch_captions).cuda()
            text_features = model.encode_text(tokens, normalize=True).cpu()
            for idx, image_id in enumerate(batch_image_ids):
                output_feature[image_id] = text_features[idx]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_feature, output_path)


if __name__ == "__main__":
    main()
