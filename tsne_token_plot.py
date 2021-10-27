'''
For a selected adversarial phrase, find the token embeddings at the selected layer.
Visualize T-SNE plot
Indicate the adversarial tokens in the plot
'''

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import sys
import os
import argparse
from models import BertSequenceClassifier
from transformers import BertTokenizer
from layer_handler import Bert_Layer_Handler
from data_loading import load_test_adapted_data_sentences

if __name__ == '__main__':

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('MODEL', type=str, help='trained .th model')
    commandLineParser.add_argument('TEST_DIR', type=str, help='attacked test data base directory')
    commandLineParser.add_argument('OUT_PATH', type=str, help='Path to save generated T-SNE plot')
    commandLineParser.add_argument('--layer_num', type=int, default=0, help="BERT layer to analyze")
    commandLineParser.add_argument('--num_points_test', type=int, default=12500, help="number of pairs data points to use test")
    commandLineParser.add_argument('--N', type=int, default=25, help="Num word substitutions used in attack")
    commandLineParser.add_argument('--iter', type=int, default=300, help="TSNE iterations")

    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/tsne_token_plot.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    np.random.seed(1)

    # Load the model
    model = BertSequenceClassifier()
    model.load_state_dict(torch.load(args.MODEL, map_location=torch.device('cpu')))
    model.eval()

    # Create model handler
    handler = Bert_Layer_Handler(model, layer_num=args.layer_num)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Load the test data
    original_list_neg, original_list_pos, attack_list_neg, attack_list_pos = load_test_adapted_data_sentences(args.TEST_DIR, args.num_points_test)
    original_list = original_list_neg + original_list_pos
    attack_list = attack_list_neg + attack_list_pos

    # Get the token embeddings for adv phrases
    original_encoded_inputs = tokenizer(original_list, padding='max_length', truncation=True, return_tensors="pt")
    original_ids = original_encoded_inputs['input_ids']

    attack_encoded_inputs = tokenizer(attack_list, padding='max_length', truncation=True, return_tensors="pt")
    attack_ids = attack_encoded_inputs['input_ids']
    attack_mask = attack_encoded_inputs['attention_mask']

    layer_embeddings = handler.get_layern_outputs(attack_ids, attack_mask)
    embeddings = torch.reshape(layer_embeddings, (attack_ids.size(0)*attack_ids.size(1), layer_embeddings.size(-1)))
    print("Embeddings", embeddings.size())

    # Use the ids to identify the adversarial embeddings - assume substituted words not in rest of phrase
    labels = torch.zeros((attack_ids.size(0), attack_ids.size(1)))
    for i in range(attack_ids.size(0)):
        for j in range(attack_ids.size(1)):
            if attack_ids[i,j] not in original_ids[i,:]:
                labels[i][j] = 1
    labels = torch.reshape(labels, (attack_ids.size(0)*attack_ids.size(1),))
    labels = ['Original' if lab==0 else 'Adversarial' for lab in labels]
    print('labels', labels.size())

    # Place into df
    feat_cols = [str(i) for i in range(embeddings.size(1))]
    df = pd.DataFrame(embeddings, columns=feat_cols)
    df['label'] = labels

    # Perform t-SNE
    data = df[feat_cols].values
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=iter)
    tsne_results = tsne.fit_transform(data)

    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="label",
        palette=sns.color_palette("bright", 4),
        data=df,
        legend="full",
        alpha=0.5
    )
    plt.savefig(args.OUT_PATH)
