from django import forms


class ImSearchForm(forms.Form):
	query = forms.ImageField(label='Upload a query sketch image')
