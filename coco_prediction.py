from coco_utils import *
from coco_train import data_transform
from coco_vocab import load_vocabulary
from coco_models import CNN, RNN
import json


def predict(args):
    # hyperparameters
    batch_size = args.batch_size

    vocab = load_vocabulary()

    test_path = 'data/test2014'
    test_file_name = 'captions_test2014_rnn152_results.json'

    val_path = 'data/val2014'
    val_file_name = 'captions_val2014_rnn152_results.json'

    test_loader = get_image_loader(dir_path=os.path.join(test_path),
                                   transform=data_transform,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=2)
    val_loader = get_image_loader(dir_path=os.path.join(val_path),
                                  transform=data_transform,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=2)

    embed_size = args.embed_size
    num_hidden = args.num_hidden

    encoder = CNN(embed_size)
    decoder = RNN(embed_size, num_hidden, len(vocab), 1)

    encoder_state_dict, decoder_state_dict, optimizer, *meta = load_model(args.checkpoint_file)
    encoder.load_state_dict(encoder_state_dict)
    decoder.load_state_dict(decoder_state_dict)

    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()

    try:
        test_results = []
        for step, (images, image_ids) in enumerate(test_loader):
            images = to_variable(images, volatile=True)
            features = encoder(images)
            captions = decoder.sample(features)
            captions = captions.cpu().data.numpy()
            captions = [translation(cap, vocab) for cap in captions]
            captions_formatted = [{'image_id': int(img_id), 'caption': cap} for img_id, cap in zip(image_ids, captions)]
            test_results.extend(captions_formatted)
            print('Sample:', captions_formatted)
    except KeyboardInterrupt:
        print('if you wish')
    finally:
        with open(test_file_name, 'w') as f:
            json.dump(test_results, f)

    try:
        val_results = []
        for step, (images, image_ids) in enumerate(val_loader):
            images = to_variable(images, volatile=True)
            features = encoder(images)
            captions = decoder.sample(features)
            captions = captions.cpu().data.numpy()
            captions = [translation(cap, vocab) for cap in captions]
            captions_formatted = [{'image_id': int(img_id), 'caption': cap} for img_id, cap in zip(image_ids, captions)]
            val_results.extend(captions_formatted)
            print('Sample:', captions_formatted)
    except KeyboardInterrupt:
        print('if you wish')
    finally:
        with open(val_file_name, 'w') as f:
            json.dump(val_results, f)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_file', type=str,
                        default=None, help='path to saved checkpoint')
    parser.add_argument('--batch_size', type=int,
                        default=128, help='size of batches')
    parser.add_argument("--test", type=int,
                        default=1, help='if eval test data')
    parser.add_argument('--embed_size', type=int,
                        default='512', help='number of embeddings')
    parser.add_argument('--num_hidden', type=int,
                        default='512', help='number of embeddings')
    args = parser.parse_args()
    predict(args)