import sys
sys.path.append('../../')

import datasets
# from datasets import load_dataset
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from torch import nn, tensor
from models import transformer

class TextDataset(nn.Module):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)
    
dataset = datasets.load_dataset('imdb', split='train')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)

tokenized_dataset = dataset.map(tokenize, batched=True)

text_dataset = TextDataset(encodings=tokenized_dataset.remove_columns(['text']),
                           labels=tokenized_dataset['label'])

loader = DataLoader(text_dataset, batch_size=32, shuffle=True)



class TransformerClassifier(nn.Module):
    def __init__(self, k, heads, num_classes, max_len, vocab_size):
        super(TransformerClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, k)
        self.transformer_block = transformer.TransformerBlock(k, heads)
        self.fc = nn.Linear(max_len * k, num_classes)

    def forward(self, input_ids, mask):
        x = self.embedding(input_ids) * mask.unsqueeze(-1)
        x = self.transformer_block(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        return logits
    

model = TransformerClassifier(k=128, heads=4, num_classes=10, max_len=128, vocab_size=len(tokenizer.vocab))



from torch.optim import Adam

optimizer = Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for batch in loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        # Forward pass
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
