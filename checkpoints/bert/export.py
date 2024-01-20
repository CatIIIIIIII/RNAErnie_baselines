import torch

content = torch.load(
    'checkpoints/bert/bert_mul_2.pth', map_location='cpu')
for name in list(content.keys()):
    if "cls." not in name:
        content[name.replace('module.', '')] = content.pop(name)
    else:
        content.pop(name)

# extend its vocab size since they don't match for [CLS] and other special token
# print(content["bert.embeddings.position_embeddings.weight"].shape)
# content["bert.embeddings.word_embeddings.weight"] = torch.cat(
#     [content["bert.embeddings.word_embeddings.weight"], torch.randn(3, 120)], dim=0)
# print(content.keys())

torch.save(content, 'checkpoints/bert/RNABERT.pth')
