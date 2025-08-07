from django.db import models
from django.contrib.auth.models import User

class AnalysisHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    input_url = models.URLField(blank=True, null=True)
    input_text = models.TextField(blank=True, null=True)
    result = models.CharField(max_length=20)  # e.g. "Fake", "Real"
    analyzed_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.result} - {self.analyzed_at.strftime('%Y-%m-%d %H:%M')}"