

import argparse

import numpy as np

from src.setup import logger
from src.models import Encoder
from src.bias_scores import weat


class WSEAT(object):

    def __init__(self, hf_model_name, n_samples=1000, parametric=False):
        self.model = None
        self.hf_model_name = hf_model_name
        self.n_samples = n_samples
        self.parametric = parametric

    def _generate_encodings(self, inp_data):
        """
        Generate encodings.
        """
        encodings = inp_data.copy()

        # load the model
        if not self.model:
            logger.info(f"Loading model {self.hf_model_name}...")            
            self.model = Encoder(self.hf_model_name, load=True)


        keys = ["targ1", "targ2", "attr1", "attr2"]

        for key in keys:
            encodings[key].update({
                "encs": self.model.encode(inp_data[key]["examples"], key),
            })

        # encodings.update({            
        #         "encs": self.model.encode(inp_data["targ1"]["examples"], "targ1"),
        #     },
        #     "targ2": {
        #         "encs": self.model.encode(inp_data["targ2"]["examples"], "targ2"),
        #     },
        #     "attr1": {
        #         "encs": self.model.encode(inp_data["attr1"]["examples"], "attr1"),
        #     },
        #     "attr2": {
        #         "encs": self.model.encode(inp_data["attr2"]["examples"], "attr2")
        #     },
        # })

        return encodings

    def run_wseat_test(self, inp_data, test, report=False):
        """
        Run the WSEAT test.
        """
        encodings = self._generate_encodings(inp_data)
        encoding_single = [e for e in encodings["targ1"]["encs"].values()][0]

        d_rep = encoding_single.size if isinstance(encoding_single, np.ndarray) else len(encoding_single)
        logger.info(f"Representation dimension: {d_rep}")

        # Run the test
        eff_size, p_value = weat.run_test(encodings, n_samples=self.n_samples, parametric=self.parametric)

        results = {
            "model": self.hf_model_name,
            "test": test,
            "p_value": p_value,
            "effect_size": eff_size,
            "num_targ1": len(encodings['targ1']["encs"]),
            "num_targ2": len(encodings['targ2']["encs"]),
            "num_attr1": len(encodings['attr1']["encs"]),
            "num_attr2": len(encodings['attr2']["encs"])
        }

        if report:
            self.report_results(results)

        return results

    def report_results(self, results):
        """
        Report the results.
        """
        logger.info("Results:")
        logger.info(results)





# # load the test data
# encs = load_json(os.path.join(args.data_dir, "%s%s" % (test, TEST_EXT)))


# # load the model
# elif model_name == ModelName.BERT.value:
# model, tokenizer = bert.load_model(args.bert_version)
# encs_targ1 = bert.encode(model, tokenizer, encs["targ1"]["examples"])
# encs_targ2 = bert.encode(model, tokenizer, encs["targ2"]["examples"])
# encs_attr1 = bert.encode(model, tokenizer, encs["attr1"]["examples"])
# encs_attr2 = bert.encode(model, tokenizer, encs["attr2"]["examples"])

# encs["targ1"]["encs"] = encs_targ1
# encs["targ2"]["encs"] = encs_targ2
# encs["attr1"]["encs"] = encs_attr1
# encs["attr2"]["encs"] = encs_attr2

# enc = [e for e in encs["targ1"]['encs'].values()][0]
# d_rep = enc.size if isinstance(enc, np.ndarray) else len(enc)


# # run the test on the encodings
# log.info("Running SEAT...")
# log.info("Representation dimension: {}".format(d_rep))
# esize, pval = weat.run_test(encs, n_samples=args.n_samples, parametric=args.parametric)
# results.append(dict(
#     model=model_name,
#     options=model_options,
#     test=test,
#     p_value=pval,
#     effect_size=esize,
#     num_targ1=len(encs['targ1']['encs']),
#     num_targ2=len(encs['targ2']['encs']),
#     num_attr1=len(encs['attr1']['encs']),
#     num_attr2=len(encs['attr2']['encs'])))