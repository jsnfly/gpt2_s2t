import torch
import random
from pathlib import Path

def extract_features_to_files(
    model, feature_extractor, examples, batch_size, max_len, output_path, val_pct
    ):
    assert all(all(key in eg for key in ['audio', 'transcription', 'id']) for eg in examples), \
        "All examples must contain keys 'audio', 'transcription, 'id'."
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    if val_pct > 0:
        (output_path / 'train').mkdir(exist_ok=True)
        (output_path / 'val').mkdir(exist_ok=True)

    model.eval().cuda()
    for i in range(0, len(examples), batch_size):
        batch = examples[i:i+batch_size]
        batch_audio = [eg['audio'] for eg in batch]
        try:
            features = feature_extractor(batch_audio, sampling_rate=16_000, return_tensors='pt', padding='max_length',
                                        max_length=max_len)
        except:
            breakpoint()

        with torch.no_grad():
            out = model(features.input_values.cuda(), attention_mask=features.attention_mask.cuda())

        assert len(batch) == len(out.last_hidden_state)
        for eg, hs in zip(batch, out.last_hidden_state.bfloat16().cpu()):
            file_path = eg['id'] + '.pt'
            if val_pct > 0:
                file_path = ('val/' if random.random() < val_pct else 'train/') + file_path
            torch.save(
                # .clone() is necessary: https://github.com/pytorch/pytorch/issues/1995
                {'transcription': eg['transcription'], 'wave2vec_features': hs.clone()},
                output_path / file_path
            )
