from django import forms
import numpy as np


class BreastCancerForm(forms.Form):
    # Características principales del dataset de cáncer de seno
    mean_radius = forms.FloatField(label='Radio medio', min_value=0)
    mean_texture = forms.FloatField(label='Textura media', min_value=0)
    mean_perimeter = forms.FloatField(label='Perímetro medio', min_value=0)
    mean_area = forms.FloatField(label='Área media', min_value=0)
    mean_smoothness = forms.FloatField(label='Suavidad media', min_value=0)

    # Puedes agregar más campos según las características del dataset


class MultipleBreastCancerForm(forms.Form):
    data = forms.CharField(
        label='Ingrese múltiples casos (uno por línea)',
        widget=forms.Textarea(attrs={'rows': 10, 'cols': 50}),
        help_text='Formato: radio_medio,textura_media,perímetro_medio,área_media,suavidad_media'
    )