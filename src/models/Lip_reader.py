import torch
from espnet.asr.asr_utils import torch_load
from espnet.asr.asr_utils import get_model_conf
from espnet.nets.lm_interface import dynamic_import_lm
from espnet.nets.scorers.length_bonus import LengthBonus
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.asr.asr_utils import add_results_to_json
import numpy as np

class Lipreading(torch.nn.Module):
    """Lipreading."""

    def __init__(self, config, odim, model, char_list, feats_position="resnet"):
        """__init__.

        :param config: ConfigParser class, contains model's configuration.
        :param feats_position: str, the position to extract features.
        """
        super(Lipreading, self).__init__()

        self.feats_position = feats_position

        self.odim = odim
        self.model = model
        self.char_list = char_list
        self.get_beam_search(config)

        self.beam_search.cuda().eval()

    def get_beam_search(self, config):
        """get_beam_search.

        :param config: ConfigParser Objects, the main configuration parser.
        """

        rnnlm = config.rnnlm
        rnnlm_conf = config.rnnlm_conf

        penalty = config.penalty
        maxlenratio = config.maxlenratio
        minlenratio = config.minlenratio
        ctc_weight = config.ctc_weight
        lm_weight = config.lm_weight
        beam_size = config.beam_size

        print(f'Beam search with ctc_weight: {ctc_weight}, lm_weight: {lm_weight}, beam_size: {beam_size}')

        sos = self.odim - 1
        eos = self.odim - 1
        scorers = self.model.scorers()

        if not rnnlm:
            lm = None
        else:
            lm_args = get_model_conf(rnnlm, rnnlm_conf)
            lm_model_module = getattr(lm_args, "model_module", "default")
            lm_class = dynamic_import_lm(lm_model_module, lm_args.backend)
            lm = lm_class(len(self.char_list), lm_args)
            torch_load(rnnlm, lm)
            print(f"load a pre-trained language model from: {rnnlm}")
            lm.eval()

        scorers["lm"] = lm
        scorers["length_bonus"] = LengthBonus(len(self.char_list))
        weights = dict(
            decoder=1.0 - ctc_weight,
            ctc=ctc_weight,
            lm=lm_weight,
            length_bonus=penalty,
        )

        # -- decoding config
        self.beam_size = beam_size
        self.nbest = 1
        self.weights = weights
        self.scorers = scorers
        self.sos = sos
        self.eos = eos
        self.ctc_weight = ctc_weight
        self.maxlenratio = maxlenratio
        self.minlenratio = minlenratio

        self.beam_search = BatchBeamSearch(
            beam_size=self.beam_size,
            vocab_size=len(self.char_list),
            weights=self.weights,
            scorers=self.scorers,
            sos=self.sos,
            eos=self.eos,
            token_list=self.char_list,
            pre_beam_score_key=None if self.ctc_weight == 1.0 else "decoder",
        )

    def predict(self, sequence, sequence_aud, search='beam'):
        """predict.

        :param sequence: ndarray, the raw sequence saved in a format of numpy array.
        """
        with torch.no_grad():
            if isinstance(sequence, np.ndarray):
                sequence = (torch.FloatTensor(sequence).cuda())
                sequence_aud = (torch.FloatTensor(sequence_aud).cuda())

            if hasattr(self.model, "module"):
                enc_feats = self.model.module.encode(sequence, sequence_aud)
            else:
                enc_feats = self.model.encode(sequence, sequence_aud)

            if search=='beam':
                nbest_hyps = self.beam_search(
                    x=enc_feats,
                    maxlenratio=self.maxlenratio,
                    minlenratio=self.minlenratio
                    )
                nbest_hyps = [
                    h.asdict() for h in nbest_hyps[:min(len(nbest_hyps), self.nbest)]
                ]

                transcription = add_results_to_json(nbest_hyps, self.char_list)

        return transcription.replace("<eos>", "")