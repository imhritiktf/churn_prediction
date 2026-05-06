from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')

    else:
        # form se data uthao → CustomData mein daalo
        data = CustomData(
            gender           = request.form.get('gender'),
            SeniorCitizen    = int(request.form.get('SeniorCitizen')),
            Partner          = request.form.get('Partner'),
            Dependents       = request.form.get('Dependents'),
            tenure           = int(request.form.get('tenure')),
            PhoneService     = request.form.get('PhoneService'),
            MultipleLines    = request.form.get('MultipleLines'),
            InternetService  = request.form.get('InternetService'),
            OnlineSecurity   = request.form.get('OnlineSecurity'),
            OnlineBackup     = request.form.get('OnlineBackup'),
            DeviceProtection = request.form.get('DeviceProtection'),
            TechSupport      = request.form.get('TechSupport'),
            StreamingTV      = request.form.get('StreamingTV'),
            StreamingMovies  = request.form.get('StreamingMovies'),
            Contract         = request.form.get('Contract'),
            PaperlessBilling = request.form.get('PaperlessBilling'),
            PaymentMethod    = request.form.get('PaymentMethod'),
            MonthlyCharges   = float(request.form.get('MonthlyCharges')),
            TotalCharges     = float(request.form.get('TotalCharges')),
        )

        df       = data.get_data_as_dataframe()
        pipeline = PredictPipeline()

        pred  = pipeline.predict(df)
        proba = pipeline.predict_proba(df)

        result     = "Will Churn ⚠️" if pred[0] == 1 else "Will Not Churn ✅"
        confidence = f"{proba[0]*100:.1f}%"

        return render_template('home.html',
                               result=result,
                               confidence=confidence)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)