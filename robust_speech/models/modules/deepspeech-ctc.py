from deepspeech_pytorch.model import DeepSpeech
from deepspeech_pytorch.loader.data_loader import ChunkSpectrogramParser
from deepspeech_pytorch.decoder import GreedyDecoder
import torch
from torchaudio.transforms import Spectrogram
from robust_speech.models.sb_hf_binding import HuggingFaceASR
import speechbrain as sb
import robust_speech as rs
import string


class DeepSpeechASR(HuggingFaceASR):
    def eval_forward(self, inputs, tokens, options, loss_options):
        model = self.modules.model
        dtype = torch.float16 if options.get("fp16", False) else torch.float32
        with torch.no_grad():
            out, lens, _ = model(*inputs)
            loss = model.criterion(out, tokens, lens, len(tokens))
            #logits = result["logits"]
            #pred_tokens = logits.argmax(dim=-1)
            decoded_output, decoded_offsets = self.hparams.decoder.decode(out, lens)
        return loss, decoded_output
    
    def train_attack_forward(self, inputs, tokens, options, loss_options):
        model = self.modules.model
        out, lens, _ = model(*inputs)
        loss = model.criterion(out, tokens, lens, len(tokens))
        decoded_output, decoded_offsets = self.hparams.decoder.decode(out, lens)
        return loss, decoded_output
    
    def compute_objectives(
        self, predictions, batch, stage, adv=False, targeted=False, reduction="mean"
    ):
        """Computes the loss (CTC+NLL) given predictions and targets."""

        loss, predicted, save_stage = predictions

        ids = batch.id

        if stage != sb.Stage.TRAIN and stage != rs.Stage.ATTACK:
            # print(pred_tokens.shape)
            # predicted_words = self.tokenizer.batch_decode(pred_tokens, skip_special_tokens=True).strip().upper().translate(
            #     str.maketrans('', '', string.punctuation))
            # predicted_words = [self.tokenizer.decode(
            #    t).strip() for t in pred_tokens]
            predicted_words = [wrd.split(" ") for wrd in predicted]
            target = [wrd.upper().translate(str.maketrans(
                '', '', string.punctuation)) for wrd in batch.wrd]
            target_words = [wrd.split(" ") for wrd in target]
            #target_words = [wrd.split(" ") for wrd in batch.wrd]
            #print(predicted_words, target_words)

            if adv:
                if targeted:
                    self.adv_wer_metric_target.append(
                        ids, predicted_words, target_words
                    )
                    self.adv_cer_metric_target.append(
                        ids, predicted, target
                    )
                    self.adv_ser_metric_target.append(
                        ids, predicted, target)
                else:
                    self.adv_wer_metric.append(
                        ids, predicted_words, target_words)
                    self.adv_cer_metric.append(
                        ids, predicted_words, target_words)
                print('adv_cer =', self.adv_cer_metric.summarize())
            else:
                self.wer_metric.append(ids, predicted_words, target_words)
                self.cer_metric.append(ids, predicted_words, target_words)
                print('cer =', self.cer_metric.summarize())
            # if adv and targeted:
                # print(" ".join(predicted_words[0]))
                #print(" ".join(target_words[0]))
        return loss

def get_deepspeech_model():    
    model = DeepSpeech.load_from_checkpoint('/ocean/projects/cis220031p/mshah1/audio_robustness_benchmark/deepspeech_ckps/librispeech_pretrained_v3.ckpt')
    return model

class DeepSpeechFeatureExtractor(object):
    def __init__(self, spec_cfg, normalize):
        self.spec = Spectrogram(
            n_fft=spec_cfg['sample_rate']*spec_cfg['window_size'],
            hop_length=spec_cfg['sample_rate']*spec_cfg['window_stride'],
            win_length=spec_cfg['sample_rate']*spec_cfg['window_size'],
            window_fn=torch.hamming_window,
        )
        self.normalize = normalize

    def __call__(self, audio):
        spec = self.spec(audio)
        spec = torch.log1p(spec)
        if self.normalize:
            spec = (spec - spec.mean()) / spec.std()
        return spec, torch.LongTensor([spec.shape[-1]])

def get_decoder(model):
    decoder = GreedyDecoder(model.labels)
    return decoder