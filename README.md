📍 Tamil Nadu Business Recommendation System
A Python-based location-aware recommendation system that helps users discover profitable business categories in Tamil Nadu. It also provides insights into local competition, average pricing, and identifies novel business opportunities based on demand and population clusters.

🚀 Features
🔍 Search local businesses by area name
🧠 Machine learning model (Random Forest) recommends business categories based on location, rating, reviews, and price
🗺️ Geolocation support using OpenStreetMap (via geopy)
🏙️ Population-based clustering to find similar business zones
💡 Suggestions for low-competition business types in the selected area
📊 Business insights with average rating and pricing for informed decisions

📁 Dataset
Place your tamil_nadu_businesses.csv file in the project directory. The dataset should include at least the following columns:
Business_Name
Category
Local_Area
Address
Latitude
Longitude
Ratings
Reviews
Price_Inr
Population

🧪 How It Works
Input your local area name
Enter desired rating, number of reviews, price range, and estimated population
The system:
Lists all businesses in the selected area
Predicts a profitable business category using a trained Random Forest Classifier
Clusters thE location based on population
Recommends novel low-competition business ideas
Shows businesses from similar areas (same cluster)

▶️ How to Run
You'll be prompted to enter:
Area (e.g., Anna Nagar, Coimbatore)
Minimum rating (0–5)
Minimum number of reviews
Target business price (₹)
Approximate population of the area
