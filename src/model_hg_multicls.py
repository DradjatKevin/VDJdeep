from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.models.bert.modeling_bert import BertModel as TransformersBertModel
from transformers.models.bert.modeling_bert import BertForMaskedLM as TransformersBertForMaskedLM
from transformers.models.bert.modeling_bert import BertForPreTraining as TransformersBertForPreTraining
from transformers.models.bert.modeling_bert import BertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput, TokenClassifierOutput


class BertPooler0(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
    
class BertPooler1(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        second_token_tensor = hidden_states[:, 1]
        pooled_output = self.dense(second_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
    
class BertPooler2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        third_token_tensor = hidden_states[:, 2]
        pooled_output = self.dense(third_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
    

class BertModel(TransformersBertModel):
    def __init__(self, config):
        super().__init__(config)

class BertForMaskedLM(TransformersBertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)

class BertForPreTraining(TransformersBertForPreTraining):
    def __init__(self, config):
        super().__init__(config)



class DNABertForSequenceAndTokenClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels_v, num_labels_d, num_labels_j):
        super().__init__(config)
        self.num_labels_v = num_labels_v
        self.num_labels_d = num_labels_d
        self.num_labels_j = num_labels_j
        self.num_Tokenlabels = 2
        self.config = config

        self.bert = BertModel(config)
        self.pooler0 = BertPooler0(config)
        self.pooler1 = BertPooler1(config)
        self.pooler2 = BertPooler2(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier_v = nn.Linear(config.hidden_size, num_labels_v)
        self.classifier_d = nn.Linear(config.hidden_size, num_labels_d)
        self.classifier_j = nn.Linear(config.hidden_size, num_labels_j)

        self.Tokenclassifier = nn.Linear(config.hidden_size, self.num_Tokenlabels) #replace 2 by more if not binary (here CDR3 or not), can easily extend to CDR1 or 2 or3 or none

        # Initialize weights and apply final processing
        #self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels_V: Optional[torch.Tensor] = None,
        labels_D: Optional[torch.Tensor] = None,
        labels_J: Optional[torch.Tensor] = None,
        Tokenlabels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # get the size of input_ids
        batch_size, seq_len = input_ids.shape
        if seq_len > 512:
            assert seq_len % 512 == 0, "seq_len should be a multiple of 512"
            # split the input_ids into multiple chunks
            input_ids = input_ids.view(-1, 512)
            attention_mask = attention_mask.view(-1, 512) if attention_mask is not None else None
            token_type_ids = token_type_ids.view(-1, 512) if token_type_ids is not None else None
            position_ids = None

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        pooled_output0 = self.pooler0(sequence_output)
        pooled_output1 = self.pooler1(sequence_output)
        pooled_output2 = self.pooler2(sequence_output)

        if seq_len > 512:
            # reshape the pooled_output
            pooled_output0 = pooled_output0.view(batch_size, -1, pooled_output0.shape[-1])
            pooled_output1 = pooled_output1.view(batch_size, -1, pooled_output1.shape[-1])
            pooled_output2 = pooled_output2.view(batch_size, -1, pooled_output2.shape[-1])
            # take the mean of the pooled_output
            pooled_output0 = torch.mean(pooled_output0, dim=1)
            pooled_output1 = torch.mean(pooled_output1, dim=1)
            pooled_output2 = torch.mean(pooled_output2, dim=1)

        pooled_output0 = self.dropout(pooled_output0)
        pooled_output1 = self.dropout(pooled_output1)
        pooled_output2 = self.dropout(pooled_output2)
        
        logits0 = self.classifier_v(pooled_output0)
        logits1 = self.classifier_d(pooled_output1)
        logits2 = self.classifier_j(pooled_output2)

        lossSeq0, lossSeq1, lossSeq2 = None, None, None
        #sequenceClassifierOutput = None

        # first CLS token
        if labels_V is not None:
            if self.config.problem_type is None:
                if self.num_labels_v == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels_v > 1 and (labels_V.dtype == torch.long or labels_V.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels_v == 1:
                    lossSeq0 = loss_fct(logits0.squeeze(), labels_V.squeeze())
                else:
                    lossSeq0 = loss_fct(logits0, labels_V)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                lossSeq0 = loss_fct(logits0.view(-1, self.num_labels_v), labels_V.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                lossSeq0 = loss_fct(logits0, labels_V)

            sequenceClassifierOutput0 = SequenceClassifierOutput(
            loss=lossSeq0,
            logits=logits0,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions)
        else :
            sequenceClassifierOutput0 = SequenceClassifierOutput(
                loss=lossSeq0,
                logits=logits0,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions
                )
            
        # second CLS token
        if labels_D is not None:
            if self.config.problem_type is None:
                if self.num_labels_d == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels_d > 1 and (labels_D.dtype == torch.long or labels_D.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels_d == 1:
                    lossSeq1 = loss_fct(logits1.squeeze(), labels_D.squeeze())
                else:
                    lossSeq1 = loss_fct(logits1, labels_D)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                lossSeq1 = loss_fct(logits1.view(-1, self.num_labels_d), labels_D.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                lossSeq1 = loss_fct(logits1, labels_D)

            sequenceClassifierOutput1 = SequenceClassifierOutput(
            loss=lossSeq1,
            logits=logits1,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions)
        else :
            sequenceClassifierOutput1 = SequenceClassifierOutput(
                loss=lossSeq1,
                logits=logits1,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions
                )
            
        # third CLS token
        if labels_J is not None:
            if self.config.problem_type is None:
                if self.num_labels_j == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels_j > 1 and (labels_J.dtype == torch.long or labels_J.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels_j == 1:
                    lossSeq2 = loss_fct(logits2.squeeze(), labels_J.squeeze())
                else:
                    lossSeq2 = loss_fct(logits2, labels_J)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                lossSeq2 = loss_fct(logits2.view(-1, self.num_labels_j), labels_J.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                lossSeq2 = loss_fct(logits2, labels_J)

            sequenceClassifierOutput2 = SequenceClassifierOutput(
            loss=lossSeq2,
            logits=logits2,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions)
        else :
            sequenceClassifierOutput2 = SequenceClassifierOutput(
                loss=lossSeq2,
                logits=logits2,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions
                )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        Tokenlogits = self.Tokenclassifier(sequence_output)
        #print(Tokenlogits.shape)

        lossToken = None
        #tokenClassifierOutput = None
        if Tokenlabels is not None:
            loss_fct = CrossEntropyLoss()
            lossToken = loss_fct(Tokenlogits.view(-1, self.num_Tokenlabels), Tokenlabels.view(-1))

            tokenClassifierOutput =  TokenClassifierOutput(
                loss=lossToken,
                logits=Tokenlogits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        else :
            tokenClassifierOutput =  TokenClassifierOutput(
                    loss=lossToken,
                    logits=Tokenlogits,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                )

        if not return_dict:
            output = (logits0, logits1, logits2, Tokenlogits,) + outputs[2:]
            return ((lossSeq0, lossSeq1, lossSeq2, lossToken,) + output) if lossSeq0 is not None else output
        # if not return_dict:
        #     output = (logits,) + outputs[2:]
        #     return ((loss,) + output) if loss is not None else output

        return sequenceClassifierOutput0, sequenceClassifierOutput1, sequenceClassifierOutput2, tokenClassifierOutput