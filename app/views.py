from django.http import HttpResponse
from django.shortcuts import render, redirect
from newspaper import Article
from django.shortcuts import render
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages
from django.contrib.auth import login, authenticate
from django.contrib.auth.decorators import login_required
from .models import AnalysisHistory
from django.core.paginator import Paginator
from django.contrib.auth.views import LoginView
import os
from django.conf import settings
from app.ML.predict_NB import analyze_text_with_lime_NB
from app.ML.predict_Bert import analyze_text_with_lime_Bert
from app.ML.predict_LR import analyze_text_with_lime_LR


def home(request):
    return render(request, "home.html")

@login_required
def submit_article(request):
    return render(request, 'submit_article.html')

@login_required 
def analyze_article(request):
    if request.method == 'POST':
        url = request.POST.get('news_url')
        text = request.POST.get('news_text')

        if url and not text:
            try:
                article = Article(url)
                article.download()
                article.parse()
                text = article.text 
            except Exception as e:
                messages.error(request, f"Failed to fetch article: {str(e)}")
                print(e)
                return render(request, 'submit_article.html')

        # Now run the fake news detection model here 
        result = analyze_text_with_lime_Bert(text)
        prediction = result["prediction"]      # "Fake" or "Real"
        confidence = result["confidence"]      # float confidence %
        top_features = result["top_features"]  # list of (feature, weight) tuples
        lime_html = load_lime_html()

        # Save to database
        AnalysisHistory.objects.create(
            user=request.user,
            input_url=url if url else None,
            input_text=text if text else None,
            result=prediction,
        )

        return render(request, 'result.html', {
            'prediction': prediction,
            'text': text,
            'lime_html': lime_html
        })

    return render(request, 'submit_article.html')

def about(request):
    load_lime_html()
    return render(request, 'about.html')

def signup_view(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'Account created successfully. You can now log in.')
            return redirect('login')  # redirect to login after successful registration
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = UserCreationForm()

    return render(request, 'registration/signup.html', {'form': form})

@login_required
def profile_view(request):
    history_list = AnalysisHistory.objects.filter(user=request.user).order_by('-analyzed_at')
    paginator = Paginator(history_list, 10)  # Show 5 records per page

    page_number = request.GET.get('page')
    history = paginator.get_page(page_number)

    return render(request, 'profile.html', {'history': history})


class CustomLoginView(LoginView):
    redirect_authenticated_user = True

    def dispatch(self, request, *args, **kwargs):
        if request.user.is_authenticated:
            return redirect(self.get_success_url())
        return super().dispatch(request, *args, **kwargs)
    
def load_lime_html():
    # Build absolute path to lime_output.html inside your app's templates
    file_path = os.path.join(
        settings.BASE_DIR, 
        'app', 'ML', 'lime_explanation.html'
    )

    with open(file_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    return html_content