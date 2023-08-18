import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the trained model from the pickle file
with open("model.pkl", "rb") as model_file:
    model_m = pickle.load(model_file)

# Assuming you have a scaler for the item vectors
# You can replace this with the actual preprocessing steps you used
def scale_item_vectors(item_vecs):
    # Your scaling logic here
    return item_vecs

# Define a function to generate movie recommendations using the loaded model
def get_recommendations(user_preferences):
    # Process user preferences and generate recommendations
    # Here, we'll assume user_preferences is a list of selected features (e.g., genre, year, etc.)
    # You'll need to preprocess user preferences accordingly

    # Convert user_preferences into a format that aligns with the model's input
    # For this example, we'll create a dummy feature vector
    user_input = np.zeros((1, 9423))  # Replace 9423 with the appropriate number of features
    # Set the relevant features based on user input, you'll need to modify this part
    for feature_index in user_preferences:
        user_input[0, feature_index] = 1.0  # Assuming user_preferences are indices of relevant features
    
    # Use the loaded model (model_m) to predict item vectors based on user input
    scaled_user_input = scale_item_vectors(user_input)  # Apply the same scaling as during training
    vm_m = model_m.predict(scaled_user_input)
    
    # Perform cosine similarity or any other relevant similarity measure
    # to find most similar items based on the generated item vector (vm_m)
    # Here, we'll return a dummy list of recommended movies
    recommendations = ["Movie A", "Movie B", "Movie C"]
    return recommendations

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Extract user preferences from the form (checkboxes, etc.)
        user_preferences = request.form.getlist("preferences")
        # Get the indices of relevant features based on user preferences
        relevant_feature_indices = [int(pref) for pref in user_preferences]
        # Get movie recommendations based on user preferences
        recommendations = get_recommendations(relevant_feature_indices)
        return render_template("index.html", recommendations=recommendations)
    return render_template("index.html", recommendations=[])

if __name__ == "__main__":
    app.run(debug=True)
