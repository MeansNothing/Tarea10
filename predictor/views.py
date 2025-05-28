from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
from .forms import BreastCancerForm, MultipleBreastCancerForm
from .ml_model import load_model
import numpy as np

model, feature_names = load_model()


def home(request):
    single_form = BreastCancerForm()
    multiple_form = MultipleBreastCancerForm()

    if request.method == 'POST':
        if 'single_submit' in request.POST:
            single_form = BreastCancerForm(request.POST)
            if single_form.is_valid():
                features = [
                    single_form.cleaned_data['mean_radius'],
                    single_form.cleaned_data['mean_texture'],
                    single_form.cleaned_data['mean_perimeter'],
                    single_form.cleaned_data['mean_area'],
                    single_form.cleaned_data['mean_smoothness'],
                    # Agrega más características según corresponda
                ]

                # Hacer predicción
                proba = model.predict_proba([features])[0]
                result = {
                    'malignant_prob': proba[0] * 100,
                    'benign_prob': proba[1] * 100,
                    'features': single_form.cleaned_data,
                    'type': 'single'
                }
                return render(request, 'predictor/results.html', {'result': result})

        elif 'multiple_submit' in request.POST:
            multiple_form = MultipleBreastCancerForm(request.POST)
            if multiple_form.is_valid():
                cases = []
                for line in multiple_form.cleaned_data['data'].split('\n'):
                    if line.strip():
                        values = [float(x.strip()) for x in line.split(',')]
                        proba = model.predict_proba([values])[0]
                        cases.append({
                            'features': values,
                            'malignant_prob': proba[0] * 100,
                            'benign_prob': proba[1] * 100
                        })

                result = {
                    'cases': cases,
                    'type': 'multiple'
                }
                return render(request, 'predictor/results.html', {'result': result})

    return render(request, 'predictor/home.html', {
        'single_form': single_form,
        'multiple_form': multiple_form
    })