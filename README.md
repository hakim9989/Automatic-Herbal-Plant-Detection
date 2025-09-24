# Automatic-Herbal-Plant-Detection
An automatic herbal plant detection system that leverages Convolutional Neural Networks (CNNs) and a Flutter + PHP backend architecture to accurately identify medicinal plants from images. This project aims to support healthcare, research, and education by providing quick and reliable identification of herbal plants along with their medicinal uses. 
# Project Overview
Identifying medicinal plants traditionally requires specialized knowledge and expertise. Misidentification can lead to health risks if toxic plants are mistaken for useful ones. This project automates plant identification by combining:
Image Upload & Detection (Flutter frontend)
Machine Learning Model (CNN-based classification)
Backend Services (PHP + MySQL for authentication, detection, and history storage)
Medicinal Plant Information (Scientific name, description, medicinal uses)
The system not only detects plants but also saves detection history for users to review later.
# Features
User Authentication (Register & Login with token-based authentication)
Plant Image Upload (via Flutter app)
Automated Plant Detection (CNN model predicts plant type + confidence score)
Plant Details (scientific name, description, medicinal uses)
History Management (view, filter, and delete past detections)
Flutter UI with responsive design and user-friendly interface
# Tech Stack
Frontend (Mobile App)
Flutter (Dart)
Shared Preferences for token storage
Backend
PHP (API services for auth, detection, history)
MySQL (User data, tokens, history storage)
Machine Learning
Convolutional Neural Networks (CNNs) for plant classification
Image preprocessing (resizing, filtering, feature extraction)
