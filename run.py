import argparse
import cv2
import json
import numpy as np
import os
import torch

from model import VitClassifier
from utils import visualize_prediction


DEFAULT_CKPT = 'logs/2023-06-1019-42-03-188079/version_0/checkpoints/last.ckpt'


def parse_args():
    parser = argparse.ArgumentParser(prog='ViT inference test')
    parser.add_argument(
        '--checkpoint', type=str,
        default=DEFAULT_CKPT,
        help='Checkpoint path',
    )
    parser.add_argument(
        '--device', type=str,
        default='cpu',
        help='Execution device',
    )
    parser.add_argument(
        '--source',
        default=0,
        help='Video stream source',
    )
    parser.add_argument(
        '--labels_map', type=str,
        default='labels.json',
        help='Path to JSON with labels map',
    )
    return parser.parse_args()


def run(args):
    net = VitClassifier.load_from_checkpoint(
        args.checkpoint,
        map_location=torch.device(args.device)
    )
    net.freeze()

    print(f'Checkpoint:\n{args.checkpoint}')
    print(f'Model:\n{net}')

    labels = None
    if os.path.isfile(args.labels_map):
        with open(args.labels_map, 'rt') as f:
            labels = json.load(f)
            labels = {int(k): v for k, v in labels.items()}

    cap = cv2.VideoCapture(args.source)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (256, 256))

        scores, classes, attention = net(
            np.expand_dims(frame, 0),
            return_scores=True
        )

        score = scores[0].numpy()
        score = '{:.3f}'.format(score)
        class_id = int(classes[0].numpy())
        if labels:
            class_id = labels[class_id]
        attention = attention[0].numpy()
        description = f'Prediction: {class_id} ({score})'

        attention_vis = visualize_prediction(
            img=frame,
            text=description,
            attention=attention,
        )
        
        print(description)
    
        cv2.imshow('Source', frame)
        cv2.imshow('Attention', attention_vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run(parse_args())
