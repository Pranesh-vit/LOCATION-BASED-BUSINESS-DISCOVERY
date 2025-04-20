import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from geopy.geocoders import Nominatim

def load_data(file_path):
    file_path = 'tamil_nadu_businesses.csv'
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip().str.title()
    return df

def get_coordinates(location_name):
    geolocator = Nominatim(user_agent="business_recommender")
    location = geolocator.geocode(location_name)
    if location:
        return location.latitude, location.longitude
    else:
        print("Location not found. Please try again.")
        return None, None

def find_businesses_in_area(df, location_name):
    location_name = location_name.lower().strip()
    df['Local_Area_Lower'] = df['Local_Area'].str.lower().fillna('')
    businesses = df[df['Local_Area_Lower'].str.contains(location_name)]
    return businesses[['Business_Name', 'Category', 'Local_Area', 'Address', 'Ratings']]

def train_recommendation_model(df):
    features = ['Latitude', 'Longitude', 'Ratings', 'Reviews', 'Price_Inr']
    label = 'Category'
    le = LabelEncoder()
    df[label] = le.fit_transform(df[label])
    category_mapping = dict(zip(le.transform(le.classes_), le.classes_))
    X = df[features]
    y = df[label]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, le, category_mapping

def recommend_business(model, le, latitude, longitude, ratings, reviews, price):
    user_input = [[latitude, longitude, ratings, reviews, price]]
    predicted_category = model.predict(user_input)
    return le.inverse_transform(predicted_category)[0]

def suggest_profitable_business(df, location_name, category):
    total = len(df[df['Category'] == category])
    avg_rating = df[df['Category'] == category]['Ratings'].mean()
    avg_price = df[df['Category'] == category]['Price_Inr'].mean()
    if total == 0:
        return f"Very few businesses in '{category}' in {location_name}, suggesting high demand with low competition."
    else:
        return (f"{location_name.title()} has {total} '{category}' businesses.\n"
                f"Avg rating: {avg_rating:.2f}, Avg price: ₹{avg_price:.2f}.\n"
                "Potential growth opportunity!")

def suggest_novel_business(df, location_name):
    location_name = location_name.lower().strip()
    df['Local_Area_Lower'] = df['Local_Area'].str.lower().fillna('')
    local_df = df[df['Local_Area_Lower'].str.contains(location_name)]
    if local_df.empty:
        return f"No businesses found in {location_name}. Try a nearby or broader area."
    category_counts = local_df['Category'].value_counts()
    median_count = category_counts.median()
    low_comp = category_counts[category_counts < median_count].index.tolist()
    if not low_comp:
        return f"{location_name.title()} has balanced business types. Review trends before investing."
    return (f"In {location_name.title()}, these business types have low competition:\n"
            + "\n".join(low_comp) + "\nConsider them for unique market entry.")

def cluster_by_local_area_and_population(df):
    if 'Local_Area' not in df.columns or 'Population' not in df.columns:
        print("Missing 'Local_Area' or 'Population'.")
        df['Cluster'] = -1
        return df

    le = LabelEncoder()
    df['Local_Area_Code'] = le.fit_transform(df['Local_Area'].astype(str))
    features = df[['Local_Area_Code', 'Population']].dropna()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    dbscan = DBSCAN(eps=0.5, min_samples=3)
    clusters = dbscan.fit_predict(scaled)

    df['Cluster'] = -1
    df.loc[features.index, 'Cluster'] = clusters
    return df, le

def get_user_cluster(df, local_area, population, le):
    local_area_code = le.transform([local_area])[0]
    features = df[['Local_Area_Code', 'Population']].dropna()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    user_scaled = scaler.transform([[local_area_code, population]])
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(scaled)
    idx = nn.kneighbors(user_scaled, return_distance=False)[0][0]
    return df.iloc[features.index[idx]]['Cluster']

# -------------------------- MAIN BLOCK ---------------------------

file_path = 'tamil_nadu_businesses.csv'
df = load_data(file_path)
df_original = df.copy()

location_name = input("Enter your area: ")
latitude, longitude = get_coordinates(location_name)
if latitude is None or longitude is None:
    exit()

rating = float(input("Minimum rating (0–5): "))
reviews = int(input("Minimum reviews: "))
price = int(input("Target price (INR): "))
population = int(input("Approximate population of area: "))

print("\nBusinesses in", location_name, ":\n", find_businesses_in_area(df, location_name))

# Train model
model, le_cat, category_mapping = train_recommendation_model(df)
recommended = recommend_business(model, le_cat, latitude, longitude, rating, reviews, price)
print("\nRecommended Business Category:", recommended)

# Cluster by local area + population
df, le_area = cluster_by_local_area_and_population(df)
user_cluster = get_user_cluster(df, location_name, population, le_area)
cluster_df = df[df['Cluster'] == user_cluster]
print(f"\nYou are in cluster: {user_cluster}. Nearby areas with similar population:\n")
print(cluster_df[['Business_Name', 'Category', 'Local_Area', 'Ratings']].head())

# Suggestions
print("\nBusiness Insight:\n", suggest_profitable_business(df, location_name, recommended))
print("\nLow Competition Business Ideas:\n", suggest_novel_business(df_original, location_name))
