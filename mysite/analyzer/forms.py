from django import forms


class TextForm(forms.Form):
    text = forms.CharField(required=True, widget=forms.Textarea(
        attrs={"class": "form-control", "placeholder": "Enter text here...", "style": "resize:none;"}
    ))
