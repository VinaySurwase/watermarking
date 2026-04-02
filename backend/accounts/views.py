from django.contrib.auth.models import User
from django.contrib.auth.hashers import make_password
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.contrib.auth import authenticate
from rest_framework_simplejwt.tokens import RefreshToken

@api_view(['POST'])
def register(request):
    data = request.data

    username = data.get('username')
    email = data.get('email')
    password = data.get('password')

    # Basic validation
    if not username or not password:
        return Response({"error": "Username and password required"}, status=400)

    if User.objects.filter(username=username).exists():
        return Response({"error": "Username already exists"}, status=400)

    # Create user manually
    user = User.objects.create(
        username=username,
        email=email,
        password=make_password(password)  
    )

    return Response({
        "message": "User registered successfully",
        "user_id": user.id
    }, status=201)
    
    
    
@api_view(['POST'])
def login(request):
    data = request.data

    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return Response({"error": "Username and password required"}, status=400)

    user = authenticate(username=username, password=password)

    if user is None:
        return Response({"error": "Invalid credentials"}, status=401)

    # Generate JWT tokens
    refresh = RefreshToken.for_user(user)

    return Response({
        "message": "Login successful",
        "refresh": str(refresh),
        "access": str(refresh.access_token),
    })