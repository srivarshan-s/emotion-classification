#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
import torch

class CustomModel(torch.nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.roberta = AutoModel.from_pretrained('roberta-base')
        self.fc = torch.nn.Linear(768, 7)

    def forward(self, ids, mask, token_type_ids):
        _, features = self.roberta(ids, attention_mask=mask,
                                   token_type_ids=token_type_ids, return_dict=False)
        output = self.fc(features)
        return output


def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mysite.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
