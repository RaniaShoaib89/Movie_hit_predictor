# Movie Hit Predictor

This Streamlit web app predicts whether a movie is likely to be a box office hit or flop based on four features like budget, popularity, vote average, and vote count.

Built using machine learning, this project compares three popular models (logistic regression, random forest, xgboost) and gives instant predictions along with visual insights like feature importance, which is represented using a barchart. The accuracy of each model is also displayed.

---

## Live Demo

🔗 [Click here to try the app](https://your-username-your-repo.streamlit.app)  
*(Replace this link after deployment)*

---

## Features

- 🎯 **Prediction Models**: Choose between Random Forest, XGBoost, and Logistic Regression
- 📊 **Feature Importance**: Visualized per model
- 📈 **Accuracy Display**: Easily compare model performances
- 🗂️ **Batch Predictions**: Upload a CSV file with movie data
- 🎨 Clean, responsive UI with background image and styling

---

##  Input Format

Whether entering values manually or uploading a CSV, the following fields are required:

- `budget`: Movie budget in USD (e.g. 100000000)
- `popularity`: Popularity metric (e.g. 22.5)
- `vote_average`: Average rating out of 10 (e.g. 7.1)
- `vote_count`: Total number of votes (e.g. 1234)

### Sample CSV

```csv
budget,popularity,vote_average,vote_count
100000000,30.5,7.8,5000
50000000,12.2,6.3,800
