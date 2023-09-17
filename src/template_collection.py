from .template import Template

class TemplateCollection(object):
    def __init__(self, templates_dict):
        self.templates = templates_dict
        self._build()

    def _build(self):
        for i, template_ in enumerate(self.templates):
            self.templates[i] = Template(template_)
    
    def __iter__(self):
        for template in self.templates:
            yield template

    def __getitem__(self, key):
        return self.templates[key]