class Template(object):

    def __init__(self, sentence) -> None:
        self.attr_placeholder = "AAA"
        self.target_placeholder = "TTT"
        self.token_sep = " "

        self.raw_sentence = sentence       

    def fill_template(self, AAA="", TTT=""):
        # return self.token_sep.join(np.flatten(list(zip(self.parts, stimuli))))
        filled_sentence = self.raw_sentence
        filled_sentence = filled_sentence.replace(self.attr_placeholder, AAA)
        filled_sentence = filled_sentence.replace(self.target_placeholder, TTT)
        return filled_sentence