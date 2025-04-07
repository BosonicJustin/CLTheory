from torch.nn.functional import normalize
import torch
from tqdm import tqdm

def extract_features(encoder, dataloader, device):
    encoder.eval()
    features = []
    labels = []

    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Extracting features"):
            x = x.to(device)
            z = encoder(x)
            z = normalize(z, dim=1)  # unit vector embeddings
            features.append(z.cpu())
            labels.append(y)

    return torch.cat(features), torch.cat(labels)


def knn_classifier(train_feats, train_labels, test_feats, test_labels, k=5):
    # Cosine similarity: higher is more similar
    sim_matrix = torch.mm(test_feats, train_feats.T)  # [num_test, num_train]
    topk = sim_matrix.topk(k=k, dim=1).indices  # [num_test, k]

    topk_labels = train_labels[topk]  # [num_test, k]
    preds = torch.mode(topk_labels, dim=1).values  # majority vote

    acc = (preds == test_labels).float().mean().item() * 100

    return acc


def run_knn_eval(encoder, dataloader_train, dataloader_test, device):
    train_feats, train_labels = extract_features(encoder, dataloader_train, device)
    test_feats, test_labels = extract_features(encoder, dataloader_test, device)

    return knn_classifier(train_feats, train_labels, test_feats, test_labels, k=5)
