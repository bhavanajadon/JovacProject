import pandas as pd
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load your trained model pipeline
pipeline = pickle.load(open('C:/Users/This PC/OneDrive/Desktop/imdb/pipeline.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form
        revenue = float(request.form['feature1'])  # Revenue (Millions)
        runtime = float(request.form['feature2'])  # Runtime (Minutes)
        genre = request.form['feature3']            # Genre
        director = request.form['feature4']         # Director
        
        # Prepare the input data for prediction as a DataFrame
        input_data = pd.DataFrame({
            'Revenue (Millions)': [revenue],
            'Runtime (Minutes)': [runtime],
            'Genre': [genre],
            'Director': [director]
        })

        # Print the input data for debugging
        print("Input Data for Prediction:\n", input_data)

        # Make prediction using the pipeline
        prediction = pipeline.predict(input_data)
        print("Prediction Result:", prediction)  # Debug line

        # Return prediction to the template
        return render_template('index.html', prediction=prediction[0])
    
    except ValueError as ve:
        print(f"ValueError: {ve}")  # Print ValueError
        return render_template('index.html', prediction="Invalid input. Please check your values.")
    
    except Exception as e:
        print(f"Error: {e}")  # Print error for debugging
        return render_template('index.html', prediction="An error occurred during prediction.")

if __name__ == '__main__':
    app.run(debug=True)
