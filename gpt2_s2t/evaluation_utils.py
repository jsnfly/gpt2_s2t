import torch
import jiwer
import numpy as np

def get_predictions(encoder_hidden_states, model, tokenizer):
    input_ids = tokenizer(
        ['Transcription:'] * encoder_hidden_states.shape[0], return_tensors='pt', padding=True
    ).input_ids.cuda()
    with torch.no_grad():
        for k in range(35):
            out = model(encoder_hidden_states, input_ids)
            input_ids = torch.cat([input_ids, out.logits.argmax(-1)[:, -1].unsqueeze(-1)], dim=-1)
    return [tokenizer.decode(ids).replace('_', '') for ids in input_ids]


def calculate_mean_loss(model, dataloader):
    losses = []
    for encoder_hidden_states, input_ids in dataloader:
        with torch.no_grad():
            out = model(encoder_hidden_states, input_ids)
        losses.append(out.loss.item())
    return np.array(losses).mean()


def calculate_wer(predictions, golds):

    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemoveWhiteSpace(replace_by_space=True),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(word_delimiter=" ")
    ])
    return jiwer.wer(golds, predictions, truth_transform=transformation, hypothesis_transform=transformation)
