import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=num_layers, 
                          dropout=dropout if num_layers > 1 else 0, 
                          batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        # src: [batch_size, src_len]
        embedded = self.dropout(self.embedding(src))
        # embedded: [batch_size, src_len, embedding_dim]
        
        outputs, hidden = self.rnn(embedded)
        # outputs: [batch_size, src_len, hidden_dim]
        # hidden: [num_layers, batch_size, hidden_dim]
        
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=num_layers,
                          dropout=dropout if num_layers > 1 else 0,
                          batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, tgt, hidden):
        # tgt: [batch_size, 1]
        # hidden: [num_layers, batch_size, hidden_dim]
        
        embedded = self.dropout(self.embedding(tgt))
        # embedded: [batch_size, 1, embedding_dim]
        
        output, hidden = self.rnn(embedded, hidden)
        # output: [batch_size, 1, hidden_dim]
        
        prediction = self.fc_out(output.squeeze(1))
        # prediction: [batch_size, vocab_size]
        
        return prediction, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, tgt, teacher_forcing_ratio=0.7):
        # src: [batch_size, src_len]
        # tgt: [batch_size, tgt_len]
        
        batch_size = tgt.shape[0]
        tgt_len = tgt.shape[1]
        tgt_vocab_size = self.decoder.fc_out.out_features
        
        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)
        
        # Get encoder outputs and hidden state
        _, hidden = self.encoder(src)
        
        # First input to the decoder is the <BOS> token
        input = tgt[:, 0:1]  # Start with BOS token
        
        for t in range(1, tgt_len):
            # Get decoder output
            output, hidden = self.decoder(input, hidden)
            
            # Store output
            outputs[:, t-1, :] = output
            
            # Decide whether to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            # Get the highest predicted token
            top1 = output.argmax(1)
            
            # Use ground truth or predicted token as next input
            input = tgt[:, t:t+1] if teacher_force else top1.unsqueeze(1)
        
        return outputs


def create_model(src_vocab_size, tgt_vocab_size, embedding_dim=256, hidden_dim=512, 
                num_layers=2, dropout=0.2, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Create a Seq2Seq model with RNN encoder and decoder
    
    Args:
        src_vocab_size: size of source vocabulary
        tgt_vocab_size: size of target vocabulary
        embedding_dim: embedding dimension
        hidden_dim: hidden dimension for RNN
        num_layers: number of RNN layers
        dropout: dropout rate
        device: device to run the model on
        
    Returns:
        model: Seq2Seq model
    """
    encoder = Encoder(src_vocab_size, embedding_dim, hidden_dim, num_layers, dropout)
    decoder = Decoder(tgt_vocab_size, embedding_dim, hidden_dim, num_layers, dropout)
    model = Seq2Seq(encoder, decoder, device)
    
    # Initialize parameters
    for name, param in model.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)
    
    return model.to(device)
