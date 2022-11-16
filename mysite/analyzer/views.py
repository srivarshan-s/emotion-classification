from django.shortcuts import render
from .forms import TextForm
from .predict import *


# Create your views here.


def home(request):

    if request.method == "POST":
        form = TextForm(request.POST)
        print(form.errors)

        if form.is_valid():
            global text
            text = form.cleaned_data.get("text")
            print(text)
            pred = get_pred(text)
            print(pred)
            url = {
                "anger": "analyzer/anger.html",
                "disgust": "analyzer/disgust.html",
                "fear": "analyzer/fear.html",
                "joy": "analyzer/joy.html",
                "sadness": "analyzer/sadness.html",
                "surprise": "analyzer/surprise.html",
                "neutral": "analyzer/neutral.html",
            }
            return render(request, url[pred])

    else:
        form = TextForm()

    context = {"form": form}
    return render(request, "analyzer/home.html", context)


def results(request):
    pass
