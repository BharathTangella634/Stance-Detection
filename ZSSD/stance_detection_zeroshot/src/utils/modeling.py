import torch
import torch.nn as nn
from transformers import BertModel, BartConfig, ModernBertModel
from transformers import BartModel, RobertaModel, DistilBertModel
from transformers import AutoModel


## ModernBert

class modern_bert_classifier(nn.Module):

    def __init__(self, num_labels, dropout):

        super(modern_bert_classifier, self).__init__()
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        self.mbert = ModernBertModel.from_pretrained("answerdotai/ModernBERT-base")
        self.mbert.pooler = None
        self.linear = nn.Linear(self.mbert.config.hidden_size*2, self.mbert.config.hidden_size)
        self.out = nn.Linear(self.mbert.config.hidden_size, num_labels)
        
    def forward(self, **kwargs):
        
        x_input_ids, x_atten_masks = kwargs['input_ids'], kwargs['attention_mask']

        last_hidden = self.mbert(input_ids=x_input_ids, attention_mask=x_atten_masks).last_hidden_state
        
        eos_token_ind = x_input_ids.eq(self.mbert.config.eos_token_id).nonzero() # tensor([[0,4],[0,5],[0,11],[1,2],[1,3],[1,6]...])
        
        

        assert len(eos_token_ind) == 2*len(kwargs['input_ids'])
        b_eos = [eos_token_ind[i][1] for i in range(len(eos_token_ind)) if i%2==0]
        e_eos = [eos_token_ind[i][1] for i in range(len(eos_token_ind)) if (i+1)%2==0]
        x_atten_clone = x_atten_masks.clone().detach()

        for begin, end, att, att2 in zip(b_eos, e_eos, x_atten_masks, x_atten_clone):
            att[begin:], att2[:begin+1] = 0, 0 # att all </s> --> 0; att2 1st and 2nd </s> --> 0
            att[0], att2[end] = 0, 0 # <s> --> 0; 3rd </s> --> 0
        
        txt_l = x_atten_masks.sum(1).to('cuda')
        topic_l = x_atten_clone.sum(1).to('cuda')
        

        txt_vec = x_atten_masks.type(torch.FloatTensor).to('cuda')
        topic_vec = x_atten_clone.type(torch.FloatTensor).to('cuda')
        txt_mean = torch.einsum('blh,bl->bh', last_hidden, txt_vec) / txt_l.unsqueeze(1)
        topic_mean = torch.einsum('blh,bl->bh', last_hidden, topic_vec) / topic_l.unsqueeze(1)

        cat = torch.cat((txt_mean, topic_mean), dim=1) #1

        
        query = self.dropout(cat)
        linear = self.relu(self.linear(query))
        out = self.out(linear)
        
        return out

# ModernBert CLS

# class modern_bert_classifier(nn.Module):
#     """
#     A ModernBERT classifier that uses the [CLS] token embedding
#     for classification (no concatenation of segment means).
#     """

#     def __init__(self, num_labels: int, dropout: float):
#         super(modern_bert_classifier, self).__init__()

#         self.mbert = ModernBertModel.from_pretrained("answerdotai/ModernBERT-base")
#         self.mbert.pooler = None  # we will grab the CLS hidden state ourselves

#         self.dropout = nn.Dropout(dropout)
#         self.relu = nn.ReLU()

#         hidden = self.mbert.config.hidden_size
#         self.linear = nn.Linear(hidden, hidden)
#         self.out = nn.Linear(hidden, num_labels)

#     def forward(self, **kwargs):
#         input_ids = kwargs["input_ids"]
#         attention_mask = kwargs["attention_mask"]

#         hidden_states = self.mbert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
#         cls_embed = hidden_states[:, 0, :]                     # [CLS] token representation

#         x = self.dropout(cls_embed)
#         x = self.relu(self.linear(x))
#         logits = self.out(x)

#         return logits

 
class bert_classifier(nn.Module):

    def __init__(self, num_labels, dropout):

        super(bert_classifier, self).__init__()
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.bert.pooler = None

        self.linear = nn.Linear(self.bert.config.hidden_size*2, self.bert.config.hidden_size) #2

        self.out = nn.Linear(self.bert.config.hidden_size, num_labels)
        
    def forward(self, **kwargs):
        
        x_input_ids, x_atten_masks, x_seg_ids = kwargs['input_ids'], kwargs['attention_mask'], kwargs['token_type_ids']
        last_hidden = self.bert(input_ids=x_input_ids, attention_mask=x_atten_masks, token_type_ids=x_seg_ids)
        
        x_atten_masks[:,0] = 0 # [CLS] --> 0 
        idx = torch.arange(0, last_hidden[0].shape[1], 1).to('cuda')
        x_seg_ind = x_seg_ids * idx
        x_att_ind = (x_atten_masks-x_seg_ids) * idx
        indices_seg = torch.argmax(x_seg_ind, 1, keepdim=True)
        indices_att = torch.argmax(x_att_ind, 1, keepdim=True)
        for seg, seg_id, att, att_id in zip(x_seg_ids, indices_seg, x_atten_masks, indices_att):
            seg[seg_id] = 0  # 2nd [SEP] --> 0 
            att[att_id:] = 0  # 1st [SEP] --> 0 
        
        txt_l = x_atten_masks.sum(1).to('cuda')
        topic_l = x_seg_ids.sum(1).to('cuda')
        txt_vec = x_atten_masks.type(torch.FloatTensor).to('cuda')
        topic_vec = x_seg_ids.type(torch.FloatTensor).to('cuda')
        txt_mean = torch.einsum('blh,bl->bh', last_hidden[0], txt_vec) / txt_l.unsqueeze(1)
        topic_mean = torch.einsum('blh,bl->bh', last_hidden[0], topic_vec) / topic_l.unsqueeze(1)
        
        cat = torch.cat((txt_mean, topic_mean), dim=1)
        
        query = self.dropout(cat)
        linear = self.relu(self.linear(query))
        out = self.out(linear)
        
        return out


# Bert CLS

# class bert_classifier(nn.Module):
#     def __init__(self, gen, model_name="bert-base-uncased", num_classes=3):
#         super(bert_classifier, self).__init__()
#         # Load the pre-trained BERT model
#         self.bert = BertModel.from_pretrained(model_name)
#         # Add two fully connected layers
#         self.fc1 = nn.Linear(self.bert.config.hidden_size, 128)  # First FC layer
#         self.relu = nn.ReLU()  # Activation
#         self.fc2 = nn.Linear(128, num_classes)  # Second FC layer for 3 output classes
#         self.dropout = nn.Dropout(0.1) if gen==0 else nn.Dropout(0.7)
#     def forward(self, **kwargs):

#         input_ids, attention_mask, token_type_ids = kwargs['input_ids'], kwargs['attention_mask'], kwargs['token_type_ids']
#         # Pass inputs through BERT
#         outputs = self.bert(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids
#         )
#         # Get the [CLS] token embedding
#         cls_output = outputs.last_hidden_state[:, 0, :]  # Using the raw [CLS] token embedding
#         # Pass through fully connected layers
#         x = self.dropout(cls_output)
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         return x


class bertweet_classifier(nn.Module):
    def __init__(self, num_labels, dropout):
        super(bertweet_classifier, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        
        self.bertweet = AutoModel.from_pretrained("vinai/bertweet-base")
        
        self.bertweet.pooler = None  
        
        self.linear = nn.Linear(self.bertweet.config.hidden_size * 2, self.bertweet.config.hidden_size)  #1
        self.out = nn.Linear(self.bertweet.config.hidden_size, num_labels)

    def forward(self, **kwargs):
        x_input_ids, x_atten_masks = kwargs["input_ids"], kwargs["attention_mask"]

        
        device = x_input_ids.device
        
        last_hidden = self.bertweet(x_input_ids, x_atten_masks).last_hidden_state
        
        eos_token_id = self.bertweet.config.eos_token_id
        eos_token_ind = x_input_ids.eq(eos_token_id).nonzero()


        # Fix assertion to prevent crashes
        assert len(eos_token_ind) == 3 * len(x_input_ids), "Mismatch in EOS token count!"

        b_eos = [eos_token_ind[i, 1].item() for i in range(len(eos_token_ind)) if i % 3 == 0]
        e_eos = [eos_token_ind[i, 1].item() for i in range(len(eos_token_ind)) if (i + 1) % 3 == 0]
        
        x_atten_clone = x_atten_masks.clone().detach()

        for begin, end, att, att2 in zip(b_eos, e_eos, x_atten_masks, x_atten_clone):
            att[begin:], att2[: begin + 2] = 0, 0  # att all </s> --> 0; att2 1st and 2nd </s> --> 0
            att[0], att2[end] = 0, 0  # <s> --> 0; 3rd </s> --> 0

        # Convert attention masks to float and ensure they are on the same device
        txt_l = x_atten_masks.sum(1).to(device)
        topic_l = x_atten_clone.sum(1).to(device)
        txt_vec = x_atten_masks.float().to(device)
        topic_vec = x_atten_clone.float().to(device)

        txt_mean = torch.einsum("blh,bl->bh", last_hidden, txt_vec) / txt_l.unsqueeze(1)
        topic_mean = torch.einsum("blh,bl->bh", last_hidden, topic_vec) / topic_l.unsqueeze(1)

        # Concatenate sentence representations
        cat = torch.cat((txt_mean, topic_mean), dim=1)

        # Apply dropout, activation, and final classification layers
        query = self.dropout(cat)
        linear = self.relu(self.linear(query))
        out = self.out(linear)

        return out


class roberta_classifier(nn.Module):
    def __init__(self, num_labels, dropout):
        super(roberta_classifier, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

       
        self.roberta = RobertaModel.from_pretrained("FacebookAI/roberta-base")
        self.roberta.pooler = None  
        
        
        self.linear = nn.Linear(self.roberta.config.hidden_size * 2, self.roberta.config.hidden_size)  #1
        self.out = nn.Linear(self.roberta.config.hidden_size, num_labels)

    def forward(self, **kwargs):
        x_input_ids, x_atten_masks = kwargs["input_ids"], kwargs["attention_mask"]

        device = x_input_ids.device
        last_hidden = self.roberta(x_input_ids, x_atten_masks).last_hidden_state
        eos_token_id = self.roberta.config.eos_token_id
        eos_token_ind = x_input_ids.eq(eos_token_id).nonzero()

    
        # Fix assertion to prevent crashes
        assert len(eos_token_ind) == 3 * len(x_input_ids), "Mismatch in EOS token count!"

        b_eos = [eos_token_ind[i, 1].item() for i in range(len(eos_token_ind)) if i % 3 == 0]
        e_eos = [eos_token_ind[i, 1].item() for i in range(len(eos_token_ind)) if (i + 1) % 3 == 0]
        
        x_atten_clone = x_atten_masks.clone().detach()

        for begin, end, att, att2 in zip(b_eos, e_eos, x_atten_masks, x_atten_clone):
            att[begin:], att2[: begin + 2] = 0, 0  # att all </s> --> 0; att2 1st and 2nd </s> --> 0
            att[0], att2[end] = 0, 0  # <s> --> 0; 3rd </s> --> 0

        # Convert attention masks to float and ensure they are on the same device
        txt_l = x_atten_masks.sum(1).to(device)
        topic_l = x_atten_clone.sum(1).to(device)
        txt_vec = x_atten_masks.float().to(device)
        topic_vec = x_atten_clone.float().to(device)

        txt_mean = torch.einsum("blh,bl->bh", last_hidden, txt_vec) / txt_l.unsqueeze(1)
        topic_mean = torch.einsum("blh,bl->bh", last_hidden, topic_vec) / topic_l.unsqueeze(1)

        # Concatenate sentence representations
        cat = torch.cat((txt_mean, topic_mean), dim=1)

        # Apply dropout, activation, and final classification layers
        query = self.dropout(cat)
        linear = self.relu(self.linear(query))
        out = self.out(linear)

        return out

class distilbert_classifier(nn.Module):
    def __init__(self, num_labels, dropout):
        super(distilbert_classifier, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        
        self.distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.distilbert.pooler = None
        
        
        self.linear = nn.Linear(self.distilbert.config.dim * 2, self.distilbert.config.dim)  #1
        self.out = nn.Linear(self.distilbert.config.dim, num_labels)

    def forward(self, **kwargs):
        x_input_ids, x_atten_masks = kwargs["input_ids"], kwargs["attention_mask"]

        device = x_input_ids.device

        last_hidden = self.distilbert(x_input_ids, x_atten_masks).last_hidden_state
        eos_token_id = 102
        
        eos_token_ind = x_input_ids.eq(eos_token_id).nonzero()

        # Fix assertion to prevent crashes
        assert len(eos_token_ind) == 2 * len(x_input_ids), "Mismatch in EOS token count!"

        b_eos = [eos_token_ind[i, 1].item() for i in range(len(eos_token_ind)) if i % 2 == 0]
        e_eos = [eos_token_ind[i, 1].item() for i in range(len(eos_token_ind)) if (i + 1) % 2 == 0]
        
        x_atten_clone = x_atten_masks.clone().detach()

        for begin, end, att, att2 in zip(b_eos, e_eos, x_atten_masks, x_atten_clone):
            att[begin:], att2[: begin + 1] = 0, 0  # att all </s> --> 0; att2 1st and 2nd </s> --> 0
            att[0], att2[end] = 0, 0  # <s> --> 0; 3rd </s> --> 0

        # Convert attention masks to float and ensure they are on the same device
        txt_l = x_atten_masks.sum(1).to(device)
        topic_l = x_atten_clone.sum(1).to(device)
        txt_vec = x_atten_masks.float().to(device)
        topic_vec = x_atten_clone.float().to(device)

        txt_mean = torch.einsum("blh,bl->bh", last_hidden, txt_vec) / txt_l.unsqueeze(1)
        topic_mean = torch.einsum("blh,bl->bh", last_hidden, topic_vec) / topic_l.unsqueeze(1)

        # Concatenate sentence representations
        cat = torch.cat((txt_mean, topic_mean), dim=1)

        # Apply dropout, activation, and final classification layers
        query = self.dropout(cat)
        linear = self.relu(self.linear(query))
        out = self.out(linear)

        return out



    


# Original bart classifier

# class bart_classifier(nn.Module):

#     def __init__(self, num_labels, dropout):

#         super(bart_classifier, self).__init__()
        
#         self.dropout = nn.Dropout(dropout)
#         self.relu = nn.ReLU()
        
#         self.config = BartConfig.from_pretrained('facebook/bart-large-mnli')
#         self.bart = BartModel.from_pretrained("facebook/bart-large-mnli")
#         self.encoder = self.bart.get_encoder()
#         self.bart.pooler = None
#         self.linear = nn.Linear(self.bart.config.hidden_size*4, self.bart.config.hidden_size) #1
#         self.out = nn.Linear(self.bart.config.hidden_size, num_labels)
        
#     def forward(self, **kwargs):
        
#         x_input_ids, x_atten_masks = kwargs['input_ids'], kwargs['attention_mask']
        
#         last_hidden = self.encoder(input_ids=x_input_ids, attention_mask=x_atten_masks).last_hidden_state
        
#         eos_token_ind = x_input_ids.eq(self.config.eos_token_id).nonzero() # tensor([[0,4],[0,5],[0,11],[1,2],[1,3],[1,6]...])
        
#         assert len(eos_token_ind) == 3*len(kwargs['input_ids'])
#         b_eos = [eos_token_ind[i][1] for i in range(len(eos_token_ind)) if i%3==0]
#         e_eos = [eos_token_ind[i][1] for i in range(len(eos_token_ind)) if (i+1)%3==0]
#         x_atten_clone = x_atten_masks.clone().detach()

#         for begin, end, att, att2 in zip(b_eos, e_eos, x_atten_masks, x_atten_clone):
#             att[begin:], att2[:begin+2] = 0, 0 # att all </s> --> 0; att2 1st and 2nd </s> --> 0
#             att[0], att2[end] = 0, 0 # <s> --> 0; 3rd </s> --> 0
        
#         txt_l = x_atten_masks.sum(1).to('cuda')
#         topic_l = x_atten_clone.sum(1).to('cuda')
#         txt_vec = x_atten_masks.type(torch.FloatTensor).to('cuda')
#         topic_vec = x_atten_clone.type(torch.FloatTensor).to('cuda')
#         txt_mean = torch.einsum('blh,bl->bh', last_hidden, txt_vec) / txt_l.unsqueeze(1)
#         topic_mean = torch.einsum('blh,bl->bh', last_hidden, topic_vec) / topic_l.unsqueeze(1)

#         # cat = torch.cat((txt_mean, topic_mean), dim=1) #1
#         cat = torch.cat((txt_mean, topic_mean, txt_mean - topic_mean, txt_mean * topic_mean), dim=1)
        
        
#         # raise Exception
#         query = self.dropout(cat)
#         linear = self.relu(self.linear(query))
#         out = self.out(linear)
        
#         return out
        

    
