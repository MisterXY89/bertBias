

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

        logger.info(f"Effect size: {eff_size}")
        logger.info(f"p-value: {p_value}")

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
