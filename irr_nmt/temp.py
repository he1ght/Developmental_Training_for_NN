from tqdm import tqdm
from emb_statistics import get_stat
from pytorch_pretrained_bert import BertModel

model = BertModel.from_pretrained('bert-base-multilingual-cased')
model.eval()
bert_emb = model.embeddings

# print(bert_emb)
vectors = []
for i, v in enumerate(tqdm(bert_emb.word_embeddings.weight)):
    vectors.append(v.detach().numpy().tolist())
word_stats = get_stat(vectors)
vectors = []
for i, v in enumerate(tqdm(bert_emb.position_embeddings.weight)):
    vectors.append(v.detach().numpy().tolist())
position_stats = get_stat(vectors)
vectors = []
for i, v in enumerate(tqdm(bert_emb.token_type_embeddings.weight)):
    vectors.append(v.detach().numpy().tolist())
token_stats = get_stat(vectors)

print("Word Embedding:")
for k, e in word_stats.items():
    print(k)
    print("{:.4f}".format(e))
print("Position Embedding:")
for k, e in position_stats.items():
    print(k)
    print("{:.4f}".format(e))
print("Token Embedding:")
for k, e in token_stats.items():
    print(k)
    print("{:.4f}".format(e))